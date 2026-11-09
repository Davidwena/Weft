#include "bj/ops/batch_suspects.hpp"
#include "bj/ops/nl.hpp"
#include "bj/ops/patterns.hpp"
#include "bj/core/multi_round.hpp"
#include "bj/proto/eq.hpp"
#include "bj/proto/remask.hpp"
#include "bj/proto/shuffle.hpp"
#include <bit>

namespace bj {
    namespace {
        std::vector<unsigned> table_widths(const InvestigationConfig &c) {
            std::vector<unsigned> w{64, 96, 0, 32, 32};
            if (c.carry_cp()) {
                w.push_back(96);
            }
            return w;
        }

        std::vector<unsigned> slot_widths(const InvestigationConfig &c) {
            std::vector<unsigned> w{64, 96, 0};
            if (c.carry_cp()) {
                w.push_back(96);
            }
            return w;
        }

        class OpenLabels final : public Step {
            const MaskedTable &table;
            unsigned party, width;
            bool bit;
            std::array<BitVec, 2> bitshares, bitopened;
            std::array<BitSlot, 2> bs;
            std::array<AlignedBuffer<std::uint32_t>, 2> shares, opened;
            std::array<RingSlot, 2> rs;
            bool bound = false, used = false, done = false;

        public:
            OpenLabels(const MaskedTable &t, unsigned p, std::size_t S)
                : table(t), party(p), width(static_cast<unsigned>(std::bit_width(S))), bit(S == 1),
                  bitshares{BitVec(bit ? t.columns[0].count : 0),
                            BitVec(bit ? t.columns[0].count : 0)},
                  bitopened{BitVec(bit ? t.columns[0].count : 0),
                            BitVec(bit ? t.columns[0].count : 0)},
                  shares{AlignedBuffer<std::uint32_t>(bit ? 0 : t.columns[0].count),
                         AlignedBuffer<std::uint32_t>(bit ? 0 : t.columns[0].count)},
                  opened{AlignedBuffer<std::uint32_t>(bit ? 0 : t.columns[0].count),
                         AlignedBuffer<std::uint32_t>(bit ? 0 : t.columns[0].count)} {
            }

            void bind(RoundPlan &p) override {
                if (bound) {
                    throw std::logic_error("label bind twice");
                }
                bound = true;
                for (unsigned d = 0; d < 2; ++d) {
                    if (bit) {
                        bs[d] = p.bits(table.columns[0].count);
                    } else {
                        rs[d] = p.ring(width, table.columns[0].count);
                    }
                }
            }

            void contribute(OpenWriter &w) override {
                if (!bound || used) {
                    throw std::logic_error("label open reused");
                }
                used = true;
                for (unsigned d = 0; d < 2; ++d) {
                    const auto &col = table.columns[3 + d];
                    if (bit) {
                        auto out = bitshares[d].words();
#pragma omp parallel for if (col.count >= 16384) schedule(static)
                        for (std::size_t word = 0; word < out.size(); ++word) {
                            std::uint64_t value = 0;
                            for (unsigned lane = 0; lane < 64 && word * 64 + lane < col.count;
                                 ++lane) {
                                auto i = word * 64 + lane;
                                value |= std::uint64_t((col.m32.data()[i] ^
                                                        (party == 0 ? col.d32.data()[i] : 0)) &
                                                       1)
                                         << lane;
                            }
                            out[word] = value;
                        }
                        w.bits(bs[d], out);
                    } else {
                        auto out = shares[d].span();
                        auto mask = native_mask<std::uint32_t>(width);
#pragma omp parallel for if (col.count >= 16384) schedule(static)
                        for (std::size_t i = 0; i < col.count; ++i) {
                            out[i] =
                                ((party == 0 ? col.d32.data()[i] : 0) - col.m32.data()[i]) & mask;
                        }
                        w.ring<std::uint32_t>(rs[d], out);
                    }
                }
            }

            void consume(OpenReader &r) override {
                if (!used || done) {
                    throw std::logic_error("label consume reused");
                }
                done = true;
                for (unsigned d = 0; d < 2; ++d) {
                    if (bit) {
                        r.bits(bs[d], bitopened[d].words());
                    } else {
                        r.ring<std::uint32_t>(rs[d], opened[d].span());
                    }
                }
            }

            std::size_t at(unsigned d, std::size_t i) const {
                return bit ? std::size_t(bitopened[d].get(i)) : opened[d].data()[i];
            }
        };

        struct Group {
            MaterialPlan view;
            MaskedTable in, out;
            AlignedBuffer<std::uint64_t> sid_delta;
            BitVec eq;
            std::array<BitVec, 2> g, sign, dsign, omega;
            std::array<AlignedBuffer<std::uint32_t>, 2> selected, prefix, kept, dvalue, dummy;
            std::unique_ptr<EqComparison<std::uint64_t>> eq_step;
            std::array<std::unique_ptr<Step>, 2> select, keep;
            std::array<std::unique_ptr<MsbFresh<std::uint32_t>>, 2> prefix_step, dummy_step;
            std::array<std::unique_ptr<FusedAnd2A<1, std::uint32_t>>, 2> b2a;
            std::unique_ptr<RemaskStep> ri, ro;
            std::unique_ptr<BandPair> band;
            BitVec negin{1}, negout{1};
            AlignedBuffer<std::uint64_t> nonoverflow{1};
            std::unique_ptr<FusedAnd2A<2, std::uint64_t>> omegastep;
            std::unique_ptr<NestedLoop> nl;
            std::uint64_t count = 0;

            Group(const InvestigationConfig &c, const MaterialPlan &p, std::size_t j)
                : view(batch_suspect_view(p, j)), in(c.kin, slot_widths(c)),
                  out(c.kout, slot_widths(c)), sid_delta(c.size()), eq(c.size()),
                  g{BitVec(c.size()), BitVec(c.size())},
                  sign{BitVec(c.size() + 1), BitVec(c.size() + 1)},
                  dsign{BitVec(c.kin), BitVec(c.kout)}, omega{BitVec(1), BitVec(1)} {
                for (unsigned d = 0; d < 2; ++d) {
                    const auto K = d ? c.kout : c.kin;
                    selected[d] = AlignedBuffer<std::uint32_t>(c.size());
                    prefix[d] = AlignedBuffer<std::uint32_t>(c.size() + 1);
                    kept[d] = AlignedBuffer<std::uint32_t>(c.size());
                    dvalue[d] = AlignedBuffer<std::uint32_t>(K);
                    dummy[d] = AlignedBuffer<std::uint32_t>(K);
                }
            }
        };

        void copy_slot(const MaskedTable &src, std::size_t i, MaskedTable &dst, std::size_t k) {
            dst.columns[0].d64.data()[k] = src.columns[0].d64.data()[i];
            dst.columns[0].m64.data()[k] = src.columns[0].m64.data()[i];
            dst.columns[1].d128.data()[k] = src.columns[1].d128.data()[i];
            dst.columns[1].m128.data()[k] = src.columns[1].m128.data()[i];
            dst.columns[2].bits.set(k, src.columns[2].bits.get(i));
            dst.columns[2].mask_bits.set(k, src.columns[2].mask_bits.get(i));
            if (dst.columns.size() > 3) {
                dst.columns[3].d128.data()[k] = src.columns[5].d128.data()[i];
                dst.columns[3].m128.data()[k] = src.columns[5].m128.data()[i];
            }
        }
    }

    struct BatchSuspects::Impl {
        InvestigationConfig c;
        std::size_t S, N, M;
        unsigned party;
        const MaterialPlan &plan;
        const InputManager &input;
        MaskedTable table;
        std::array<BitVec, 2> directions;
        std::vector<std::unique_ptr<Group>> groups;
        std::vector<std::vector<Step *>> layers{11};
        std::vector<std::unique_ptr<RoundExecutor>> rounds;
        std::unique_ptr<RemaskStep> relabel;
        std::unique_ptr<Shuffle> shuffle;
        std::unique_ptr<OpenLabels> labels;
        std::vector<std::vector<Step *>> joinlayers{2};
        std::vector<std::unique_ptr<RoundExecutor>> joinrounds;
        std::unique_ptr<MultiRound> scheduler;
        std::vector<StageStats> stages;
        std::vector<std::array<std::size_t, 2>> sizes;
        bool used = false;

        Impl(const InvestigationConfig &config,
             std::size_t suspects,
             MaterialStore &store,
             const MaterialPlan &p,
             const InputManager &in)
            : c(config), S(suspects), N(c.size()), M(N + S * (c.kin + c.kout)),
              party(store.party()), plan(p), input(in), table(M, table_widths(c)),
              directions{BitVec(N), BitVec(N)}, sizes(S) {
            if (compile_batch_suspects(c, S).fingerprint() != p.fingerprint()) {
                throw std::invalid_argument("batch material mismatch");
            }
            if (c.algorithm != "nl") {
                throw std::invalid_argument("X11 batch join requires NL");
            }
            for (unsigned i = 0; i < 4; ++i) {
                stages.emplace_back(c.n);
            }
            for (std::size_t j = 0; j < S; ++j) {
                auto g = std::make_unique<Group>(c, p, j);
                auto &v = g->view;
                g->eq_step =
                    std::make_unique<EqComparison<std::uint64_t>>(store,
                                                                  v,
                                                                  named_comparison(v, "compact.eq"),
                                                                  0,
                                                                  g->sid_delta.span(),
                                                                  g->eq.words());
                layers[0].push_back(g->eq_step.get());
                for (unsigned d = 0; d < 2; ++d) {
                    std::string label = d ? "out" : "in";
                    g->select[d] = std::make_unique<FusedAnd2A<2, std::uint32_t>>(
                        store,
                        v,
                        named_gate(v, "compact.select." + label),
                        std::array<std::span<const std::uint64_t>, 2>{g->eq.words(),
                                                                      directions[d].words()},
                        g->selected[d].span());
                    layers[1].push_back(g->select[d].get());
                    g->prefix_step[d] = std::make_unique<MsbFresh<std::uint32_t>>(
                        store,
                        v,
                        named_comparison(v, "compact.prefix." + label),
                        g->prefix[d].span(),
                        g->sign[d].words());
                    layers[2].push_back(&g->prefix_step[d]->opening());
                    layers[3].push_back(&g->prefix_step[d]->comparison());
                    g->keep[d] = std::make_unique<FusedAnd2A<3, std::uint32_t>>(
                        store,
                        v,
                        named_gate(v, "compact.keep." + label),
                        std::array<std::span<const std::uint64_t>, 3>{
                            g->eq.words(), directions[d].words(), g->g[d].words()},
                        g->kept[d].span());
                    layers[4].push_back(g->keep[d].get());
                    g->dummy_step[d] = std::make_unique<MsbFresh<std::uint32_t>>(
                        store,
                        v,
                        named_comparison(v, "compact.dummy." + label),
                        g->dvalue[d].span(),
                        g->dsign[d].words());
                    layers[5].push_back(&g->dummy_step[d]->opening());
                    layers[6].push_back(&g->dummy_step[d]->comparison());
                    g->b2a[d] = std::make_unique<FusedAnd2A<1, std::uint32_t>>(
                        store,
                        v,
                        named_gate(v, "compact.dummy_b2a." + label),
                        std::array<std::span<const std::uint64_t>, 1>{g->dsign[d].words()},
                        g->dummy[d].span());
                    layers[7].push_back(g->b2a[d].get());
                }
                std::vector<std::size_t> im{named_mask(v, "slot.tin"),
                                            named_mask(v, "slot.ain"),
                                            named_mask(v, "slot.vin")},
                    om{named_mask(v, "slot.tout"),
                       named_mask(v, "slot.aout"),
                       named_mask(v, "slot.vout")};
                if (c.carry_cp()) {
                    im.push_back(named_mask(v, "slot.cpin"));
                    om.push_back(named_mask(v, "slot.cpout"));
                }
                g->ri = std::make_unique<RemaskStep>(g->in, store, v, im);
                g->ro = std::make_unique<RemaskStep>(g->out, store, v, om);
                layers[10].push_back(g->ri.get());
                layers[10].push_back(g->ro.get());
                if (c.pattern == "P1") {
                    g->band = make_band_pair(c, store, v, g->in, g->out);
                    if (!g->band->opening_steps().empty()) {
                        throw std::invalid_argument("batch P1 requires C1+C2");
                    }
                    g->omegastep = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
                        store,
                        v,
                        named_gate(v, "output.omega"),
                        std::array<std::span<const std::uint64_t>, 2>{g->negin.words(),
                                                                      g->negout.words()},
                        g->nonoverflow.span());
                    auto a = g->band->comparison_steps(), b = g->band->product_steps();
                    joinlayers[0].insert(joinlayers[0].end(), a.begin(), a.end());
                    joinlayers[1].insert(joinlayers[1].end(), b.begin(), b.end());
                    joinlayers[1].push_back(g->omegastep.get());
                } else {
                    g->nl = std::make_unique<NestedLoop>(
                        c, store, v, g->in, g->out, g->omega[0].words(), g->omega[1].words());
                }
                groups.push_back(std::move(g));
                if (S > 1) {
                    std::vector<std::size_t> discard;
                    const auto &layout = store.layout();
                    const auto pre = "suspect" + std::to_string(j) + ".";
                    for (std::size_t i = 0; i < p.comparisons.size(); ++i) {
                        if (p.comparisons[i].name.starts_with(pre)) {
                            const auto &spec = layout.comparisons[i];
                            discard.insert(discard.end(), spec.tables.begin(), spec.tables.end());
                            discard.insert(discard.end(), spec.aux.begin(), spec.aux.end());
                            for (auto &eval : spec.evaluations) {
                                discard.insert(discard.end(), eval.begin(), eval.end());
                            }
                        }
                    }
                    for (std::size_t i = 0; i < p.gates.size(); ++i) {
                        if (p.gates[i].name.starts_with(pre)) {
                            discard.push_back(layout.gate_masks[i]);
                            discard.insert(discard.end(),
                                           layout.gate_products[i].begin(),
                                           layout.gate_products[i].end());
                        }
                    }
                    for (std::size_t i = 0; i < p.masks.size(); ++i) {
                        if (p.masks[i].name.starts_with(pre) && layout.masks[i] != SIZE_MAX) {
                            discard.push_back(layout.masks[i]);
                        }
                    }
                    for (std::size_t i = 0; i < p.bitmuls.size(); ++i) {
                        if (p.bitmuls[i].name.starts_with(pre)) {
                            for (auto id : layout.bitmuls[i]) {
                                discard.push_back(id);
                            }
                        }
                    }
                    store.release_decoded_segments(discard);
                }
            }
            const std::array<std::size_t, 2> mask{named_mask(p, "table.fin"),
                                                  named_mask(p, "table.fout")};
            const std::array<unsigned, 2> cols{3, 4};
            relabel = std::make_unique<RemaskStep>(table, store, p, mask, cols);
            layers[8].push_back(relabel.get());
            shuffle = std::make_unique<Shuffle>(table, store, p, 0);
            labels = std::make_unique<OpenLabels>(table, party, S);
            layers[9].push_back(labels.get());
            for (auto &layer : layers) {
                rounds.push_back(std::make_unique<RoundExecutor>(layer));
            }
            if (c.pattern == "P1") {
                for (auto &layer : joinlayers) {
                    joinrounds.push_back(std::make_unique<RoundExecutor>(layer));
                }
            }
        }

        void prepare() {
            std::size_t offset = 0;
            for (unsigned b = 0; b < c.n; ++b) {
                auto suf = "_" + std::to_string(b);
                const auto &sid = input.at(named_mask(plan, "sid" + suf));
                const auto &dir = input.at(named_mask(plan, "dir" + suf));
                const auto &t = input.at(named_mask(plan, "t" + suf));
                const auto &a = input.at(named_mask(plan, "a" + suf));
#pragma omp parallel for if (c.rows[b] >= 16384) schedule(static)
                for (std::size_t i = 0; i < c.rows[b]; ++i) {
                    auto r = offset + i;
                    table.columns[0].d64.data()[r] = t.delta64.data()[i];
                    table.columns[0].m64.data()[r] = t.mask64.data()[i];
                    table.columns[1].d128.data()[r] = a.delta128.data()[i];
                    table.columns[1].m128.data()[r] = a.mask128.data()[i];
                }
                for (std::size_t i = 0; i < c.rows[b]; ++i) {
                    directions[0].set(offset + i, dir.share_bits.get(i) ^ (party == 0));
                    directions[1].set(offset + i, dir.share_bits.get(i));
                    table.columns[2].bits.set(offset + i, true);
                }
                if (c.carry_cp()) {
                    const auto &cp = input.at(named_mask(plan, "cp" + suf));
                    for (std::size_t i = 0; i < c.rows[b]; ++i) {
                        table.columns[5].d128.data()[offset + i] = cp.delta128.data()[i];
                        table.columns[5].m128.data()[offset + i] = cp.mask128.data()[i];
                    }
                }
                for (auto &g : groups) {
                    auto subject = input.at(named_mask(g->view, "subject")).delta64.data()[0];
#pragma omp parallel for if (c.rows[b] >= 16384) schedule(static)
                    for (std::size_t i = 0; i < c.rows[b]; ++i) {
                        g->sid_delta.data()[offset + i] = sid.delta64.data()[i] - subject;
                    }
                }
                offset += c.rows[b];
            }
            const auto theta = std::uint64_t{1} << c.bt;
            for (std::size_t j = 0; j < S; ++j) {
                auto base = N + j * (c.kin + c.kout);
                for (std::size_t k = 0; k < c.kin; ++k) {
                    table.columns[0].d64.data()[base + k] = theta;
                }
                for (std::size_t k = 0; k < c.kout; ++k) {
                    table.columns[0].d64.data()[base + c.kin + k] = 0 - theta;
                }
            }
        }

        void prefixes() {
            for (auto &g : groups) {
                for (unsigned d = 0; d < 2; ++d) {
                    std::uint32_t sum = 0;
                    auto K = d ? c.kout : c.kin;
                    for (std::size_t i = 0; i < N; ++i) {
                        sum += g->selected[d].data()[i];
                        g->prefix[d].data()[i] =
                            (sum - (party == 0 ? static_cast<std::uint32_t>(K + 1) : 0)) &
                            native_mask<std::uint32_t>(c.kp);
                    }
                    g->prefix[d].data()[N] =
                        (sum - (party == 0 ? static_cast<std::uint32_t>(K + 1) : 0)) &
                        native_mask<std::uint32_t>(c.kp);
                }
            }
        }

        void truncated() {
            for (auto &g : groups) {
                for (unsigned d = 0; d < 2; ++d) {
                    for (std::size_t w = 0; w < g->g[d].words().size(); ++w) {
                        g->g[d].words()[w] =
                            g->sign[d].words()[w] & low_mask64(static_cast<unsigned>(
                                                        std::min<std::size_t>(64, N - w * 64)));
                    }
                    g->omega[d].set(0, g->sign[d].get(N) ^ (party == 0));
                }
            }
        }

        void dummies() {
            for (auto &g : groups) {
                for (unsigned d = 0; d < 2; ++d) {
                    std::uint32_t sum = 0;
                    for (auto v : g->kept[d].span()) {
                        sum += v;
                    }
                    auto K = d ? c.kout : c.kin;
                    for (std::size_t k = 0; k < K; ++k) {
                        g->dvalue[d].data()[k] =
                            (sum + (party == 0 ? static_cast<std::uint32_t>(k - K) : 0)) &
                            native_mask<std::uint32_t>(c.kp);
                    }
                }
            }
        }

        void assemble() {
            for (unsigned d = 0; d < 2; ++d) {
                auto &col = table.columns[3 + d];
                col.dealer_knows_mask = false;

#pragma omp parallel for if (N >= 16384) schedule(static)
                for (std::size_t i = 0; i < N; ++i) {
                    std::uint32_t value = 0;
                    for (std::size_t j = 0; j < S; ++j) {
                        value -= static_cast<std::uint32_t>(j + 1) * groups[j]->kept[d].data()[i];
                    }
                    col.m32.data()[i] = value;
                }
                auto K = d ? c.kout : c.kin;
                for (std::size_t j = 0; j < S; ++j) {
                    auto base = N + j * (c.kin + c.kout) + (d ? c.kin : 0);
                    for (std::size_t k = 0; k < K; ++k) {
                        col.m32.data()[base + k] =
                            0 - static_cast<std::uint32_t>(j + 1) * groups[j]->dummy[d].data()[k];
                    }
                }
            }
        }

        void gather() {
            for (std::size_t i = 0; i < M; ++i) {
                auto a = labels->at(0, i), b = labels->at(1, i);
                if (a > S || b > S || (a && b)) {
                    throw std::runtime_error("invalid batch labels or direction overlap");
                }
                for (unsigned d = 0; d < 2; ++d) {
                    auto label = d ? b : a;
                    if (!label) {
                        continue;
                    }
                    auto j = label - 1;
                    auto &dst = d ? groups[j]->out : groups[j]->in;
                    auto k = sizes[j][d]++;
                    if (k >= dst.columns[0].count) {
                        throw std::runtime_error("too many slots in group");
                    }
                    copy_slot(table, i, dst, k);
                }
            }
            for (std::size_t j = 0; j < S; ++j) {
                if (sizes[j][0] != c.kin || sizes[j][1] != c.kout) {
                    throw std::runtime_error("fixed-length batch marker invariant failed");
                }
                groups[j]->in.mark_public_reordered();
                groups[j]->out.mark_public_reordered();
            }
        }

        void capture(Net &net, unsigned j, Timer &timer) {
            StageStats now(c.n);
            now.capture(net, timer.seconds());
            stages[j].seconds = now.seconds;
            StageStats before(c.n);
            for (unsigned k = 0; k < j; ++k) {
                before.open.rounds += stages[k].open.rounds;
                before.open.hops += stages[k].open.hops;
                before.open.contribution_bits += stages[k].open.contribution_bits;
                for (unsigned b = 0; b < c.n; ++b) {
                    before.peers[b].sent_payload += stages[k].peers[b].sent_payload;
                    before.peers[b].sent_wire += stages[k].peers[b].sent_wire;
                    before.peers[b].received_payload += stages[k].peers[b].received_payload;
                    before.peers[b].received_wire += stages[k].peers[b].received_wire;
                }
            }
            stages[j].capture(net, now.seconds, &before);
            timer.reset();
        }

        void run(Net &net) {
            if (used) {
                throw std::logic_error("batch reused");
            }
            used = true;
            net.reset_stats();
            Timer timer;
            prepare();
            rounds[0]->run(net);
            rounds[1]->run(net);
            capture(net, 0, timer);
            prefixes();
            rounds[2]->run(net);
            rounds[3]->run(net);
            truncated();
            rounds[4]->run(net);
            dummies();
            rounds[5]->run(net);
            rounds[6]->run(net);
            rounds[7]->run(net);
            capture(net, 1, timer);
            assemble();
            rounds[8]->run(net);
            shuffle->run(net);
            rounds[9]->run(net);
            gather();
            capture(net, 2, timer);
            rounds[10]->run(net);
            capture(net, 3, timer);
            net.reset_stats();
            if (c.pattern == "P1") {
                for (auto &g : groups) {
                    g->band->prepare();
                    g->negin.words()[0] = g->omega[0].words()[0] ^ (party == 0);
                    g->negout.words()[0] = g->omega[1].words()[0] ^ (party == 0);
                }
                joinrounds[0]->run(net);
                for (auto &g : groups) {
                    g->band->combine();
                }
                joinrounds[1]->run(net);
                for (auto &g : groups) {
                    std::uint64_t sum = 0;
                    auto values = g->band->shares(0);
#pragma omp parallel for reduction(+ : sum) if (values.size() >= 16384) schedule(static)
                    for (std::size_t i = 0; i < values.size(); ++i) {
                        sum += values[i];
                    }
                    g->count = sum;
                }
            } else {
                if (!scheduler) {
                    throw std::logic_error("pattern scheduler must be prepared before online");
                }
                scheduler->run();
            }
        }
    };

    BatchSuspects::BatchSuspects(const InvestigationConfig &c,
                                 std::size_t S,
                                 MaterialStore &s,
                                 const MaterialPlan &p,
                                 const InputManager &in)
        : impl_(std::make_unique<Impl>(c, S, s, p, in)) {
    }

    BatchSuspects::~BatchSuspects() = default;

    void BatchSuspects::prepare(Net &net, unsigned threads) {
        if (impl_->c.pattern == "P1") {
            return;
        }
        std::vector<ScheduledJob> jobs;
        for (auto &g : impl_->groups) {
            auto *ptr = g.get();
            jobs.push_back({[ptr, &net]() {
                                ptr->nl->run(net);
                            },
                            {}});
        }
        impl_->scheduler =
            std::make_unique<MultiRound>(net,
                                         std::move(jobs),
                                         impl_->S * (impl_->c.pairs() * 128 + 1048576),
                                         512 * impl_->S,
                                         threads,
                                         true);
    }

    void BatchSuspects::run(Net &n) {

        impl_->run(n);
    }

    std::vector<std::uint64_t> BatchSuspects::count_shares() const {
        std::vector<std::uint64_t> out;
        for (auto &g : impl_->groups) {
            out.push_back(g->nl ? g->nl->count_shares()[0] : g->count);
        }
        return out;
    }

    std::vector<std::uint64_t> BatchSuspects::overflow_shares() const {
        std::vector<std::uint64_t> out;
        for (auto &g : impl_->groups) {
            out.push_back(g->nl ? g->nl->overflow_share()
                                : (impl_->party == 0 ? 1u : 0u) - g->nonoverflow.data()[0]);
        }
        return out;
    }

    const std::vector<StageStats> &BatchSuspects::stages() const {
        return impl_->stages;
    }

    const std::vector<std::array<std::size_t, 2>> &BatchSuspects::slot_counts() const {
        return impl_->sizes;
    }
}
