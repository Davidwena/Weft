#include "bj/ops/compact.hpp"
#include "bj/proto/eq.hpp"
#include "bj/proto/msb.hpp"
#include "bj/proto/and2a.hpp"
#include "bj/proto/remask.hpp"
#include "bj/proto/shuffle.hpp"

namespace bj {
    namespace {
        std::vector<unsigned> table_widths(const InvestigationConfig &c) {
            std::vector<unsigned> w{64, 96, 0, 32, 32};
            if (c.carry_cp()) {
                w.push_back(96);
            }
            for (const auto &f : c.filters) {
                (void)f;
                w.push_back(0);
            }
            return w;
        }

        std::vector<unsigned> slot_widths(const InvestigationConfig &c) {
            std::vector<unsigned> w{64, 96, 0};
            if (c.carry_cp()) {
                w.push_back(96);
            }
            for (const auto &f : c.filters) {
                (void)f;
                w.push_back(0);
            }
            return w;
        }

        class OpenFlags final : public Step {
            const MaskedTable &table_;
            unsigned party_;
            std::array<BitVec, 2> shares_, opened_;
            std::array<BitSlot, 2> slots_;
            bool bound_ = false, used_ = false, done_ = false;

        public:
            OpenFlags(const MaskedTable &table, unsigned party)
                : table_(table), party_(party),
                  shares_{BitVec(table.columns[0].count), BitVec(table.columns[0].count)},
                  opened_{BitVec(table.columns[0].count), BitVec(table.columns[0].count)} {
            }

            void bind(RoundPlan &p) override {
                if (bound_) {
                    throw std::logic_error("flags bound twice");
                }
                bound_ = true;
                for (auto &slot : slots_) {
                    slot = p.bits(table_.columns[0].count);
                }
            }

            void contribute(OpenWriter &w) override {
                if (!bound_ || used_) {
                    throw std::logic_error("flags open reused");
                }
                used_ = true;

                for (unsigned j = 0; j < 2; ++j) {
                    const auto &col = table_.columns[3 + j];
                    auto bits = shares_[j].words();
#pragma omp parallel for if (col.count >= 16384) schedule(static)
                    for (std::size_t word = 0; word < bits.size(); ++word) {
                        std::uint64_t v = 0;
                        for (unsigned lane = 0; lane < 64 && word * 64 + lane < col.count; ++lane) {
                            const auto i = word * 64 + lane;
                            v |=
                                std::uint64_t(
                                    (col.m32.data()[i] ^ (party_ == 0 ? col.d32.data()[i] : 0)) & 1)
                                << lane;
                        }
                        bits[word] = v;
                    }
                    w.bits(slots_[j], bits);
                }
            }

            void consume(OpenReader &r) override {
                if (!used_ || done_) {
                    throw std::logic_error("flags consume state");
                }
                done_ = true;
                for (unsigned j = 0; j < 2; ++j) {
                    r.bits(slots_[j], opened_[j].words());
                }
            }

            const BitVec &at(unsigned j) const {
                return opened_.at(j);
            }
        };
    }

    struct Compact::Impl {
        InvestigationConfig c;
        unsigned party;
        const MaterialPlan &plan;
        const InputManager *input;
        const MaskedTable *candidates;
        std::size_t candidate_in;
        std::array<std::uint32_t, 2> prior, cumulative{};
        std::size_t N, M;
        MaskedTable table, in, out;
        AlignedBuffer<std::uint64_t> sid_delta;
        BitVec eq;
        std::array<BitVec, 2> directions, g, keep_sign, dummy_sign, omega;
        std::array<AlignedBuffer<std::uint32_t>, 2> selected, prefix, kept, dummy_value, dummy;
        std::unique_ptr<EqComparison<std::uint64_t>> eq_step;
        std::array<std::unique_ptr<Step>, 2> select_step;
        std::array<std::unique_ptr<MsbFresh<std::uint32_t>>, 2> prefix_step, dummy_step;
        std::array<std::unique_ptr<Step>, 2> keep_step;
        std::array<std::unique_ptr<FusedAnd2A<1, std::uint32_t>>, 2> dummy_b2a;
        std::unique_ptr<RemaskStep> remask_flags, remask_in, remask_out;
        std::unique_ptr<Shuffle> shuffle;
        std::unique_ptr<OpenFlags> flags;
        std::vector<std::vector<Step *>> layers;
        std::vector<std::unique_ptr<RoundExecutor>> rounds;
        CompactStats stats;
        bool used = false;
        std::vector<std::array<std::size_t, 4>> input_ids;
        std::size_t subject_id = 0;
        std::vector<std::vector<std::size_t>> filter_ids;

        Impl(const InvestigationConfig &config,
             MaterialStore &material,
             const MaterialPlan &p,
             const InputManager *inputs,
             std::span<const std::uint64_t> row_union,
             std::array<std::uint32_t, 2> previous,
             const MaskedTable *candidate = nullptr,
             std::size_t incoming = 0)
            : c(config), party(material.party()), plan(p), input(inputs), candidates(candidate),
              candidate_in(incoming), prior(previous), N(c.size()), M(N + c.kin + c.kout),
              table(M, table_widths(c)), in(c.kin, slot_widths(c)), out(c.kout, slot_widths(c)),
              sid_delta(N), eq(N), directions{BitVec(N), BitVec(N)}, g{BitVec(N), BitVec(N)},
              keep_sign{BitVec(N + 1), BitVec(N + 1)}, dummy_sign{BitVec(c.kin), BitVec(c.kout)},
              omega{BitVec(1), BitVec(1)}, layers(11) {
            subject_id = named_mask(p, "subject");
            for (unsigned bank = 0; bank < c.n; ++bank) {
                const auto suffix = "_" + std::to_string(bank);
                input_ids.push_back({named_mask(p, "sid" + suffix),
                                     named_mask(p, "dir" + suffix),
                                     named_mask(p, "t" + suffix),
                                     named_mask(p, "a" + suffix)});
            }
            for (const auto &f : c.filters) {
                filter_ids.emplace_back();
                for (unsigned bank = 0; bank < c.n; ++bank) {
                    filter_ids.back().push_back(
                        named_mask(p, "f_" + f + "_" + std::to_string(bank)));
                }
            }
            if (!candidates) {
                eq_step =
                    std::make_unique<EqComparison<std::uint64_t>>(material,
                                                                  p,
                                                                  named_comparison(p, "compact.eq"),
                                                                  0,
                                                                  sid_delta.span(),
                                                                  eq.words());
                layers[0].push_back(eq_step.get());
            }
            for (unsigned d = 0; d < 2; ++d) {
                const std::string label = d ? "out" : "in";
                const auto K = d ? c.kout : c.kin;
                selected[d] = AlignedBuffer<std::uint32_t>(N);
                prefix[d] = AlignedBuffer<std::uint32_t>(N + 1);
                kept[d] = AlignedBuffer<std::uint32_t>(N);
                dummy_value[d] = AlignedBuffer<std::uint32_t>(K);
                dummy[d] = AlignedBuffer<std::uint32_t>(K);
                if (c.filters.empty()) {
                    select_step[d] = std::make_unique<FusedAnd2A<2, std::uint32_t>>(
                        material,
                        p,
                        named_gate(p, "compact.select." + label),
                        std::array<std::span<const std::uint64_t>, 2>{eq.words(),
                                                                      directions[d].words()},
                        selected[d].span());
                } else {
                    if (row_union.size() != bits_words(N)) {
                        throw std::invalid_argument("compact requires secret filter union");
                    }
                    select_step[d] = std::make_unique<FusedAnd2A<3, std::uint32_t>>(
                        material,
                        p,
                        named_gate(p, "compact.select." + label),
                        std::array<std::span<const std::uint64_t>, 3>{
                            eq.words(), directions[d].words(), row_union},
                        selected[d].span());
                }
                layers[1].push_back(select_step[d].get());
                prefix_step[d] = std::make_unique<MsbFresh<std::uint32_t>>(
                    material,
                    p,
                    named_comparison(p, "compact.prefix." + label),
                    prefix[d].span(),
                    keep_sign[d].words());
                layers[2].push_back(&prefix_step[d]->opening());
                layers[3].push_back(&prefix_step[d]->comparison());
                if (c.filters.empty()) {
                    keep_step[d] = std::make_unique<FusedAnd2A<3, std::uint32_t>>(
                        material,
                        p,
                        named_gate(p, "compact.keep." + label),
                        std::array<std::span<const std::uint64_t>, 3>{
                            eq.words(), directions[d].words(), g[d].words()},
                        kept[d].span());
                } else {
                    keep_step[d] = std::make_unique<FusedAnd2A<4, std::uint32_t>>(
                        material,
                        p,
                        named_gate(p, "compact.keep." + label),
                        std::array<std::span<const std::uint64_t>, 4>{
                            eq.words(), directions[d].words(), g[d].words(), row_union},
                        kept[d].span());
                }
                layers[4].push_back(keep_step[d].get());
                dummy_step[d] = std::make_unique<MsbFresh<std::uint32_t>>(
                    material,
                    p,
                    named_comparison(p, "compact.dummy." + label),
                    dummy_value[d].span(),
                    dummy_sign[d].words());
                layers[5].push_back(&dummy_step[d]->opening());
                layers[6].push_back(&dummy_step[d]->comparison());
                dummy_b2a[d] = std::make_unique<FusedAnd2A<1, std::uint32_t>>(
                    material,
                    p,
                    named_gate(p, "compact.dummy_b2a." + label),
                    std::array<std::span<const std::uint64_t>, 1>{dummy_sign[d].words()},
                    dummy[d].span());
                layers[7].push_back(dummy_b2a[d].get());
            }
            std::vector<std::size_t> flag_masks{named_mask(p, "table.fin"),
                                                named_mask(p, "table.fout")};
            std::vector<unsigned> columns{3, 4};
            if (candidates) {
                if (!c.filters.empty() || candidate_in > N ||
                    candidates->columns.size() != slot_widths(c).size() ||
                    candidates->columns[0].count != N) {
                    throw std::invalid_argument("candidate compact shape");
                }
                for (auto [name, column] : std::vector<std::pair<std::string, unsigned>>{
                         {"time", 0}, {"amount", 1}, {"vld", 2}}) {
                    flag_masks.push_back(named_mask(p, "table." + name));
                    columns.push_back(column);
                }
                if (c.carry_cp()) {
                    flag_masks.push_back(named_mask(p, "table.cp"));
                    columns.push_back(5);
                }
            }
            remask_flags = std::make_unique<RemaskStep>(table, material, p, flag_masks, columns);
            layers[8].push_back(remask_flags.get());
            shuffle = std::make_unique<Shuffle>(table, material, p, 0);
            stats.shuffle_bytes = shuffle->bytes();
            flags = std::make_unique<OpenFlags>(table, party);
            layers[9].push_back(flags.get());
            std::vector<std::size_t> im{
                named_mask(p, "slot.tin"), named_mask(p, "slot.ain"), named_mask(p, "slot.vin")},
                om{named_mask(p, "slot.tout"),
                   named_mask(p, "slot.aout"),
                   named_mask(p, "slot.vout")};
            if (c.carry_cp()) {
                im.push_back(named_mask(p, "slot.cpin"));
                om.push_back(named_mask(p, "slot.cpout"));
            }
            for (const auto &f : c.filters) {
                im.push_back(named_mask(p, "slot.f_" + f + "_in"));
                om.push_back(named_mask(p, "slot.f_" + f + "_out"));
            }
            remask_in = std::make_unique<RemaskStep>(in, material, p, im);
            remask_out = std::make_unique<RemaskStep>(out, material, p, om);
            layers[10] = {remask_in.get(), remask_out.get()};
            for (auto &layer : layers) {
                rounds.push_back(std::make_unique<RoundExecutor>(layer));
            }
        }

        void prepare_input() {
            if (candidates) {
                for (std::size_t row = 0; row < N; ++row) {
                    for (unsigned j = 0; j < candidates->columns.size(); ++j) {
                        const auto &src = candidates->columns[j];
                        auto &dst = table.columns[j == 3 ? 5 : j];
                        if (!src.width) {
                            dst.bits.set(row, src.bits.get(row));
                            dst.mask_bits.set(row, src.mask_bits.get(row));
                        } else if (src.width <= 64) {
                            dst.d64.data()[row] = src.d64.data()[row];
                            dst.m64.data()[row] = src.m64.data()[row];
                        } else {
                            dst.d128.data()[row] = src.d128.data()[row];
                            dst.m128.data()[row] = src.m128.data()[row];
                        }
                    }
                    const auto &v = candidates->columns[2];
                    eq.set(row, v.mask_bits.get(row) ^ (party == 0 && v.bits.get(row)));
                    directions[row < candidate_in ? 0 : 1].set(row, party == 0);
                }
                const auto theta = std::uint64_t{1} << c.bt;
                for (std::size_t j = 0; j < c.kin; ++j) {
                    table.columns[0].d64.data()[N + j] = theta;
                }
                for (std::size_t j = 0; j < c.kout; ++j) {
                    table.columns[0].d64.data()[N + c.kin + j] = 0 - theta;
                }
                table.mark_public_reordered();
                return;
            }

            const auto subject = input->at(subject_id).delta64.data()[0];
            std::size_t offset = 0;
            for (unsigned bank = 0; bank < c.n; ++bank) {
                const auto &sid = input->at(input_ids[bank][0]);
                const auto &direction = input->at(input_ids[bank][1]);
                const auto &t = input->at(input_ids[bank][2]);
                const auto &a = input->at(input_ids[bank][3]);
#pragma omp parallel for if (c.rows[bank] >= 16384) schedule(static)
                for (std::size_t i = 0; i < c.rows[bank]; ++i) {
                    const auto row = offset + i;
                    sid_delta.data()[row] = sid.delta64.data()[i] - subject;
                    table.columns[0].d64.data()[row] = t.delta64.data()[i];
                    table.columns[0].m64.data()[row] = t.mask64.data()[i];
                    table.columns[1].d128.data()[row] = a.delta128.data()[i];
                    table.columns[1].m128.data()[row] = a.mask128.data()[i];
                }
                for (std::size_t i = 0; i < c.rows[bank]; ++i) {
                    const auto bit = direction.share_bits.get(i);
                    directions[0].set(offset + i, bit ^ (party == 0));
                    directions[1].set(offset + i, bit);
                }
                if (c.carry_cp()) {
                    const auto &cp = input->at(named_mask(plan, "cp_" + std::to_string(bank)));
                    for (std::size_t i = 0; i < c.rows[bank]; ++i) {
                        table.columns[5].d128.data()[offset + i] = cp.delta128.data()[i];
                        table.columns[5].m128.data()[offset + i] = cp.mask128.data()[i];
                    }
                }
                for (std::size_t j = 0; j < c.filters.size(); ++j) {
                    const auto &col = input->at(filter_ids[j][bank]);
                    auto &target = table.columns[5 + (c.carry_cp() ? 1 : 0) + j];
                    for (std::size_t i = 0; i < c.rows[bank]; ++i) {
                        target.bits.set(offset + i, col.delta_bits.get(i));
                        target.mask_bits.set(offset + i, col.mask_bits.get(i));
                    }
                }
                offset += c.rows[bank];
            }
            for (std::size_t i = 0; i < N; ++i) {
                table.columns[2].bits.set(i, true);
            }
            const auto theta = std::uint64_t{1} << c.bt;
            for (std::size_t j = 0; j < c.kin; ++j) {
                table.columns[0].d64.data()[N + j] = theta;
            }
            for (std::size_t j = 0; j < c.kout; ++j) {
                table.columns[0].d64.data()[N + c.kin + j] = 0 - theta;
            }
        }

        void prefix_values() {
            for (unsigned d = 0; d < 2; ++d) {
                const auto K = d ? c.kout : c.kin;
                std::uint32_t sum = prior[d];
                for (std::size_t i = 0; i < N; ++i) {
                    sum += selected[d].data()[i];
                    prefix[d].data()[i] =
                        (sum - (party == 0 ? static_cast<std::uint32_t>(K + 1) : 0)) &
                        static_cast<std::uint32_t>(width_mask(c.kp));
                }
                cumulative[d] = sum;
                prefix[d].data()[N] = (sum - (party == 0 ? static_cast<std::uint32_t>(K + 1) : 0)) &
                                      static_cast<std::uint32_t>(width_mask(c.kp));
            }
        }

        void truncate_bits() {
            for (unsigned d = 0; d < 2; ++d) {
                for (std::size_t w = 0; w < g[d].words().size(); ++w) {
                    g[d].words()[w] =
                        keep_sign[d].words()[w] &
                        low_mask64(static_cast<unsigned>(std::min<std::size_t>(64, N - w * 64)));
                }
                omega[d].set(0, keep_sign[d].get(N) ^ (party == 0));
            }
        }

        void dummy_values() {
            for (unsigned d = 0; d < 2; ++d) {
                std::uint32_t sum = 0;
                for (auto value : kept[d].span()) {
                    sum += value;
                }
                const auto K = d ? c.kout : c.kin;
                for (std::size_t j = 0; j < K; ++j) {
                    dummy_value[d].data()[j] =
                        (sum + (party == 0 ? static_cast<std::uint32_t>(j - K) : 0)) &
                        static_cast<std::uint32_t>(width_mask(c.kp));
                }
            }
        }

        void assemble_flags() {
            for (unsigned d = 0; d < 2; ++d) {
                auto &column = table.columns[3 + d];
                column.dealer_knows_mask = false;
                for (std::size_t i = 0; i < N; ++i) {
                    column.m32.data()[i] = 0 - kept[d].data()[i];
                }
                const auto K = d ? c.kout : c.kin, base = N + (d ? c.kin : 0);
                for (std::size_t j = 0; j < K; ++j) {
                    column.m32.data()[base + j] = 0 - dummy[d].data()[j];
                }
            }
        }

        void gather_slots() {
            std::array<std::size_t, 2> sizes{};
            for (std::size_t i = 0; i < M; ++i) {
                for (unsigned d = 0; d < 2; ++d) {
                    if (flags->at(d).get(i)) {
                        auto &target = d ? out : in;
                        auto dst = sizes[d]++;
                        if (dst >= target.columns[0].count) {
                            continue;
                        }
                        target.columns[0].d64.data()[dst] = table.columns[0].d64.data()[i];
                        target.columns[0].m64.data()[dst] = table.columns[0].m64.data()[i];
                        target.columns[1].d128.data()[dst] = table.columns[1].d128.data()[i];
                        target.columns[1].m128.data()[dst] = table.columns[1].m128.data()[i];
                        target.columns[2].bits.set(dst, table.columns[2].bits.get(i));
                        target.columns[2].mask_bits.set(dst, table.columns[2].mask_bits.get(i));
                        if (c.carry_cp()) {
                            target.columns[3].d128.data()[dst] = table.columns[5].d128.data()[i];
                            target.columns[3].m128.data()[dst] = table.columns[5].m128.data()[i];
                        }
                        for (std::size_t j = 0; j < c.filters.size(); ++j) {
                            auto &col = target.columns[3 + (c.carry_cp() ? 1 : 0) + j];
                            const auto &from = table.columns[5 + (c.carry_cp() ? 1 : 0) + j];
                            col.bits.set(dst, from.bits.get(i));
                            col.mask_bits.set(dst, from.mask_bits.get(i));
                        }
                    }
                }
            }
            if (sizes[0] != c.kin || sizes[1] != c.kout) {
                throw std::runtime_error("fixed-length marker invariant failed");
            }
            for (std::size_t w = 0; w < flags->at(0).words().size(); ++w) {
                if (flags->at(0).words()[w] & flags->at(1).words()[w]) {
                    throw std::runtime_error("direction marker overlap");
                }
            }
            in.mark_public_reordered();
            out.mark_public_reordered();
        }

        void run(Net &net) {
            if (used) {
                throw std::logic_error("compact already executed");
            }
            used = true;
            auto start = net.open_stats();
            Timer timer;
            prepare_input();
            if (!candidates) {
                rounds[0]->run(net);
            }
            stats.seconds[0] = timer.seconds();
            timer.reset();
            rounds[1]->run(net);
            stats.seconds[1] = timer.seconds();
            timer.reset();
            prefix_values();
            rounds[2]->run(net);
            rounds[3]->run(net);
            truncate_bits();
            rounds[4]->run(net);
            stats.seconds[2] = timer.seconds();
            timer.reset();
            dummy_values();
            rounds[5]->run(net);
            rounds[6]->run(net);
            rounds[7]->run(net);
            stats.seconds[3] = timer.seconds();
            timer.reset();
            assemble_flags();
            rounds[8]->run(net);
            stats.seconds[4] = timer.seconds();
            timer.reset();
            shuffle->run(net);
            stats.seconds[5] = timer.seconds();
            timer.reset();
            rounds[9]->run(net);
            gather_slots();
            stats.seconds[6] = timer.seconds();
            timer.reset();
            rounds[10]->run(net);
            stats.seconds[7] = timer.seconds();
            const auto end = net.open_stats();
            stats.open_rounds = end.rounds - start.rounds;
            stats.hops = end.hops - start.hops;
            stats.contribution_bits = end.contribution_bits - start.contribution_bits;
            stats.logical_rounds = stats.open_rounds + c.n;
        }
    };

    Compact::Compact(const InvestigationConfig &c,
                     MaterialStore &s,
                     const MaterialPlan &p,
                     const InputManager &in,
                     std::span<const std::uint64_t> row_union,
                     std::array<std::uint32_t, 2> prior)
        : impl_(std::make_unique<Impl>(c, s, p, &in, row_union, prior)) {
    }

    Compact::Compact(const InvestigationConfig &c,
                     MaterialStore &s,
                     const MaterialPlan &p,
                     const MaskedTable &candidates,
                     std::size_t incoming)
        : impl_(std::make_unique<Impl>(c,
                                       s,
                                       p,
                                       nullptr,
                                       std::span<const std::uint64_t>{},
                                       std::array<std::uint32_t, 2>{},
                                       &candidates,
                                       incoming)) {
    }

    std::array<std::uint32_t, 2> Compact::cumulative_count_shares() const {
        return impl_->cumulative;
    }

    MaterialPlan compile_candidate_compact(const InvestigationConfig &c) {
        auto p = compile_investigation(c);
        for (auto &mask : p.masks) {
            if (mask.name == "table.time" || mask.name == "table.amount" ||
                mask.name == "table.vld" || mask.name == "table.cp") {
                mask.fresh = true;
                mask.distribute = true;
                mask.terms.clear();
            }
        }
        p.validate();
        return p;
    }

    Compact::~Compact() = default;

    void Compact::evaluate_local() {
        if (impl_->eq_step) {
            impl_->eq_step->evaluate_local(impl_->eq.words());
        }
    }

    void Compact::run(Net &n) {
        impl_->run(n);
    }

    const MaskedTable &Compact::inbound() const {
        return impl_->in;
    }

    const MaskedTable &Compact::outbound() const {
        return impl_->out;
    }

    std::span<const std::uint64_t> Compact::omega_in() const {
        return impl_->omega[0].words();
    }

    std::span<const std::uint64_t> Compact::omega_out() const {
        return impl_->omega[1].words();
    }

    const CompactStats &Compact::stats() const {
        return impl_->stats;
    }
}
