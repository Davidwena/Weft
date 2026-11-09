#include "bj/ops/masked_sort.hpp"
#include "bj/ops/patterns.hpp"
#include "bj/ops/query_post.hpp"
#include "bj/proto/eq.hpp"
#include "bj/proto/interval.hpp"
#include "bj/proto/multiply.hpp"
#include <numeric>

namespace bj {
    namespace {
        template <class T> T share(const MaskedColumn &c, std::size_t i, unsigned party) {
            const auto d = c.width <= 32   ? uint128_t(c.d32.data()[i])
                           : c.width <= 64 ? uint128_t(c.d64.data()[i])
                                           : c.d128.data()[i];
            const auto m = c.width <= 32   ? uint128_t(c.m32.data()[i])
                           : c.width <= 64 ? uint128_t(c.m64.data()[i])
                                           : c.m128.data()[i];
            return static_cast<T>((party == 0 ? d : uint128_t{0}) - m);
        }

        template <class T> void assign_share(MaskedColumn &c, std::size_t i, T value) {
            if (c.width <= 32) {
                c.d32.data()[i] = 0;
                c.m32.data()[i] = 0u - static_cast<std::uint32_t>(value);
            } else if (c.width <= 64) {
                c.d64.data()[i] = 0;
                c.m64.data()[i] = 0ull - static_cast<std::uint64_t>(value);
            } else {
                c.d128.data()[i] = 0;
                c.m128.data()[i] = (uint128_t{0} - static_cast<uint128_t>(value)) &
                                   native_mask<uint128_t>(c.width);
            }
        }

        void copy_at(const MaskedColumn &a, std::size_t i, MaskedColumn &b, std::size_t j) {
            if (a.width != b.width) {
                throw std::invalid_argument("pattern column width mismatch");
            }
            if (!a.width) {
                b.bits.set(j, a.bits.get(i));
                b.mask_bits.set(j, a.mask_bits.get(i));
            } else if (a.width <= 32) {
                b.d32.data()[j] = a.d32.data()[i];
                b.m32.data()[j] = a.m32.data()[i];
            } else if (a.width <= 64) {
                b.d64.data()[j] = a.d64.data()[i];
                b.m64.data()[j] = a.m64.data()[i];
            } else {
                b.d128.data()[j] = a.d128.data()[i];
                b.m128.data()[j] = a.m128.data()[i];
            }
        }

        void boolean_share(const MaskedColumn &c, BitVec &out, unsigned party) {
            for (std::size_t w = 0; w < out.words().size(); ++w) {
                out.words()[w] = c.mask_bits.words()[w] ^ (party == 0 ? c.bits.words()[w] : 0);
            }
        }

        std::size_t multiply_id(const MaterialPlan &p, const std::string &name) {
            for (std::size_t i = 0; i < p.multiplications.size(); ++i) {
                if (p.multiplications[i].name == name) {
                    return i;
                }
            }
            throw std::invalid_argument("unknown pattern triple");
        }

        std::vector<unsigned> amount_widths(unsigned h, unsigned money, bool degree) {
            std::vector<unsigned> widths(5 + h, 0);
            widths[0] = money;
            widths[1] = 96;
            if (degree) {
                widths.push_back(64);
            }
            return widths;
        }

        std::vector<unsigned> level_widths(unsigned ell, bool degree) {
            std::vector<unsigned> widths(4 + ell, 0);
            widths[0] = 32;
            if (degree) {
                widths.push_back(96);
                widths.push_back(64);
            }
            return widths;
        }

        struct DegreeLevel {
            unsigned ell, party, h, markers;
            std::size_t N;
            bool degree;
            MaskedTable &source;
            DegreeLevel *previous;
            BitVec pt, ps, ng, b, nb;
            AlignedBuffer<std::uint64_t> u, pos, neg, b64, v, S, product;
            AlignedBuffer<std::uint32_t> b32, Z, factor, dst_product;
            std::array<std::unique_ptr<FusedAnd2A<2, std::uint64_t>>, 3> ands;
            std::unique_ptr<FusedAnd2A<1, std::uint32_t>> small_b;
            std::unique_ptr<FusedAnd2A<1, std::uint64_t>> wide_b;
            std::unique_ptr<BeaverProduct<std::uint64_t>> product_step;
            std::unique_ptr<BeaverProduct<std::uint32_t>> destination_step;
            std::unique_ptr<MaskedTable> work, next;
            std::unique_ptr<RemaskStep> remask;
            std::unique_ptr<Shuffle> shuffle;
            std::unique_ptr<RevealDestination> reveal;
            AlignedBuffer<std::size_t> inverse;
            AlignedBuffer<unsigned char> seen;
            std::vector<unsigned> columns;

            DegreeLevel(unsigned level,
                        unsigned run,
                        unsigned bits,
                        bool collect,
                        MaterialStore &store,
                        const MaterialPlan &plan,
                        MaskedTable &src,
                        DegreeLevel *prior)
                : ell(level), party(store.party()), h(bits), markers(level == bits - 1 ? 2 : 1),
                  N(src.columns[0].count), degree(collect), source(src), previous(prior), pt(N),
                  ps(N), ng(N), b(N), nb(N), u(N), pos(N), neg(N), b64(collect ? N : 0), v(N), S(N),
                  product(N), b32(N), Z(N), factor(N), dst_product(N), inverse(level ? N : 0),
                  seen(level ? N : 0) {
                const auto name = "pdc.level." + std::to_string(level) + "." + std::to_string(run);
                const std::array<std::array<std::span<const std::uint64_t>, 2>, 3> inputs{
                    {{pt.words(), nb.words()}, {ps.words(), b.words()}, {ng.words(), b.words()}}};
                const std::array<std::span<std::uint64_t>, 3> outputs{
                    u.span(), pos.span(), neg.span()};
                for (unsigned j = 0; j < 3; ++j) {
                    ands[j] = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
                        store,
                        plan,
                        named_gate(plan, name + ".and." + std::to_string(j)),
                        inputs[j],
                        outputs[j]);
                }
                small_b = std::make_unique<FusedAnd2A<1, std::uint32_t>>(
                    store,
                    plan,
                    named_gate(plan, name + ".b32"),
                    std::array<std::span<const std::uint64_t>, 1>{b.words()},
                    b32.span());
                if (degree) {
                    wide_b = std::make_unique<FusedAnd2A<1, std::uint64_t>>(
                        store,
                        plan,
                        named_gate(plan, name + ".b64"),
                        std::array<std::span<const std::uint64_t>, 1>{b.words()},
                        b64.span());
                }
                product_step = std::make_unique<BeaverProduct<std::uint64_t>>(
                    store,
                    plan,
                    multiply_id(plan, name + ".mul64"),
                    degree ? b64.span() : v.span(),
                    S.span(),
                    product.span());
                if (ell) {
                    destination_step = std::make_unique<BeaverProduct<std::uint32_t>>(
                        store,
                        plan,
                        multiply_id(plan, name + ".mul32"),
                        b32.span(),
                        factor.span(),
                        dst_product.span());
                    auto widths = level_widths(ell, degree);
                    work = std::make_unique<MaskedTable>(previous ? 0 : N, widths);
                    next = std::make_unique<MaskedTable>(previous ? 0 : N, widths);
                    for (auto *table : {work.get(), next.get()}) {
                        for (auto &col : table->columns) {
                            col.count = N;
                        }
                    }
                    std::vector<std::size_t> masks;
                    for (std::size_t j = 0; j < widths.size(); ++j) {
                        masks.push_back(named_mask(plan, name + ".mask." + std::to_string(j)));
                        columns.push_back(static_cast<unsigned>(j));
                    }
                    remask = std::make_unique<RemaskStep>(*work, store, plan, masks);
                    shuffle =
                        std::make_unique<Shuffle>(*work, store, plan, named_shuffle(plan, name));
                    reveal = std::make_unique<RevealDestination>(work->columns[0], party);
                }
            }

            unsigned source_id() const {
                return ell == h - 1 ? 1u : 5 + ell;
            }

            unsigned source_acc() const {
                return ell == h - 1 ? 5 + h : 6 + ell;
            }

            std::uint64_t accumulated(std::size_t i) const {
                return share<std::uint64_t>(source.columns[source_acc()], i, party) +
                       product.data()[i];
            }

            void prepare() {
                boolean_share(source.columns[markers], pt, party);
                boolean_share(source.columns[markers + 1], ps, party);
                boolean_share(source.columns[markers + 2], ng, party);
                boolean_share(source.columns[markers + 3 + ell], b, party);
                for (std::size_t w = 0; w < nb.words().size(); ++w) {
                    nb.words()[w] =
                        b.words()[w] ^ (party == 0 ? low_mask64(static_cast<unsigned>(
                                                         std::min<std::size_t>(64, N - w * 64)))
                                                   : 0);
                }
            }

            void prefix() {
                const auto half = std::size_t{1} << ell, block = 2 * half;
#pragma omp parallel for if (N >= 16384) schedule(static)
                for (std::size_t blockno = 0; blockno < (N + block - 1) / block; ++blockno) {
                    const auto start = blockno * block, end = std::min(N, start + block),
                               zeros = std::min(half, end - start);
                    std::uint64_t sum = 0;
                    std::uint32_t z = 0;
                    for (std::size_t i = start; i < end; ++i) {
                        v.data()[i] = pos.data()[i] - neg.data()[i];
                        S.data()[i] = sum;
                        sum += u.data()[i];
                        Z.data()[i] = z;
                        factor.data()[i] =
                            (party == 0 ? static_cast<std::uint32_t>(zeros + i - start) : 0u) -
                            2 * z;
                        z += (party == 0 ? 1u : 0u) - b32.data()[i];
                    }
                }
            }

            void destinations() {
                if (!ell) {
                    return;
                }
                for (std::size_t j = 0; previous && j < work->columns.size(); ++j) {
                    const auto old = j < 4 + ell ? j : j + 1;
                    work->columns[j] = std::move(previous->work->columns[old]);
                }
                const auto block = std::size_t{1} << (ell + 1);
                for (std::size_t i = 0; i < N; ++i) {
                    assign_share(
                        work->columns[0],
                        i,
                        (party == 0 ? static_cast<std::uint32_t>((i / block) * block) : 0u) +
                            Z.data()[i] + dst_product.data()[i]);
                    for (unsigned j = 0; j < 3 + ell; ++j) {
                        copy_at(source.columns[markers + j], i, work->columns[1 + j], i);
                    }
                    if (degree) {
                        copy_at(source.columns[source_id()], i, work->columns[4 + ell], i);
                        assign_share(work->columns[5 + ell], i, accumulated(i));
                    }
                }
                work->mark_public_reordered();
                for (std::size_t j = 0; previous && j < next->columns.size(); ++j) {
                    const auto old = j < 4 + ell ? j : j + 1;
                    next->columns[j] = std::move(previous->next->columns[old]);
                }
            }

            void scatter() {
                std::fill(seen.span().begin(), seen.span().end(), 0);
                const auto dst = reveal->positions();
                for (std::size_t i = 0; i < N; ++i) {
                    if (dst[i] >= N || seen.data()[dst[i]]++) {
                        throw std::runtime_error("pattern destination is not a permutation");
                    }
                    inverse.data()[dst[i]] = i;
                }
                gather_columns(*work, *next, columns, inverse.span());
            }
        };

        struct Layer {
            std::array<std::unique_ptr<DegreeLevel>, 2> runs;
            std::vector<Step *> a, b, c, d;
            std::unique_ptr<RoundExecutor> ar, br, cr, dr;
            std::unique_ptr<ShuffleGroup> shuffles;
        };

        class PatternDC final : public PatternOperator {
            InvestigationConfig c_;
            unsigned party_, h_, money_;
            std::size_t Kx_, Ky_, N_;
            const MaskedTable &x_;
            const MaskedTable &y_;
            bool empty_, used_ = false;
            BitVec amount_, valid_, adjacent_, boundary_;
            AlignedBuffer<std::uint64_t> amount_delta_, weight_, difference_;
            AlignedBuffer<uint128_t> group_;
            std::unique_ptr<QueryPost> post_;
            std::unique_ptr<Interval<std::uint64_t>> amount_step_;
            std::unique_ptr<FusedAnd2A<2, std::uint64_t>> weight_step_;
            std::unique_ptr<EqFresh<std::uint64_t>> adjacent_step_;
            std::unique_ptr<FusedAnd2A<1, uint128_t>> boundary_step_;
            std::vector<Step *> a_, b_, eqopen_, eqcmp_, boundary_steps_, empty_steps_;
            std::unique_ptr<RoundExecutor> ar_, br_, eqor_, eqcr_, boundary_round_, empty_round_;
            std::unique_ptr<MaskedTable> initial_, group_input_;
            std::array<std::unique_ptr<MaskedTable>, 2> money_input_, restore_input_;
            std::unique_ptr<MaskedSortGroup<std::uint64_t>> time_sort_, restore_sort_;
            std::unique_ptr<MaskedSortGroup<uint128_t>> group_sort_, money_sort_;
            std::vector<Layer> levels_;
            DCStats stats_;

            template <class T>
            std::unique_ptr<MaskedSortGroup<T>> sorts(std::span<MaskedTable *const> tables,
                                                      MaterialStore &s,
                                                      const MaterialPlan &p,
                                                      std::initializer_list<std::string> names,
                                                      unsigned width) {
                return std::make_unique<MaskedSortGroup<T>>(
                    tables, s, p, std::span<const std::string>(names.begin(), names.size()), width);
            }

            void record_sort(const auto &sort) {
                stats_.shuffle_rounds += c_.n;
                stats_.shuffle_seconds += sort.shuffle_seconds;
            }

        public:
            PatternDC(const InvestigationConfig &c,
                      MaterialStore &s,
                      const MaterialPlan &p,
                      const MaskedTable &in,
                      const MaskedTable &out,
                      std::span<const std::uint64_t> oi,
                      std::span<const std::uint64_t> oo,
                      std::span<const std::uint64_t> query_filter)
                : c_(c), party_(s.party()),
                  h_(static_cast<unsigned>(std::bit_width(
                      c.x_size() + 2 * c.y_size() ? c.x_size() + 2 * c.y_size() - 1 : 0))),
                  money_(72), Kx_(c.x_size()), Ky_(c.y_size()), N_(Kx_ + 2 * Ky_),
                  x_(c.pattern == "P4" ? out : in),
                  y_(c.pattern == "P2" || c.pattern == "P1" ? out : in),
                  empty_(c.pattern == "P3" ? N_ == 0 : Kx_ == 0 || Ky_ == 0), amount_(c.kin),
                  valid_(c.kin), adjacent_(c.pattern == "P4" && !empty_ ? N_ - 1 : 0),
                  boundary_(c.pattern == "P4" && !empty_ ? N_ : 0), amount_delta_(c.kin),
                  weight_(c.kin), difference_(adjacent_.size()), group_(boundary_.size()) {
                post_ = std::make_unique<QueryPost>(c,
                                                    s,
                                                    p,
                                                    y_,
                                                    oi,
                                                    oo,
                                                    amount_.words(),
                                                    valid_.words(),
                                                    query_filter,
                                                    c.pattern == "P3" && query_filter.empty()
                                                        ? weight_.span()
                                                        : std::span<const std::uint64_t>{});
                if (empty_) {
                    empty_steps_ = {&post_->omega_step()};
                    empty_round_ = std::make_unique<RoundExecutor>(empty_steps_);
                    return;
                }
                if (c.pattern == "P3") {
                    const std::array<unsigned, 3> widths{64, 96, 64};
                    initial_ = std::make_unique<MaskedTable>(N_, widths);
                    std::array<MaskedTable *, 1> tables{initial_.get()};
                    time_sort_ = sorts<std::uint64_t>(tables, s, p, {"pdc.time"}, 64);
                    const std::array<unsigned, 2> rw{96, 64};
                    restore_input_[0] = std::make_unique<MaskedTable>(N_, rw);
                    tables[0] = restore_input_[0].get();
                    restore_sort_ =
                        sorts<std::uint64_t>(tables, s, p, {"pdc.restore"}, std::max(2u, h_ + 1));
                    const std::array<BandWindow, 1> window{{{c.amount_lower, c.amount_upper - 1}}};
                    const std::array<std::span<std::uint64_t>, 1> outputs{amount_.words()};
                    amount_step_ = std::make_unique<Interval<std::uint64_t>>(
                        s,
                        p,
                        named_comparison(p, "pattern.amount.interval"),
                        amount_delta_.span(),
                        window,
                        outputs);
                    weight_step_ = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
                        s,
                        p,
                        named_gate(p, "pdc.weight"),
                        std::array<std::span<const std::uint64_t>, 2>{amount_.words(),
                                                                      valid_.words()},
                        weight_.span());
                    a_ = {amount_step_.get()};
                    b_ = {weight_step_.get(), &post_->omega_step()};
                    ar_ = std::make_unique<RoundExecutor>(a_);
                    br_ = std::make_unique<RoundExecutor>(b_);
                    return;
                }
                std::vector<unsigned> iw{64, 96, 96, 96, 0, 0, 0};
                if (c.pattern == "P4") {
                    iw.push_back(96);
                }
                initial_ = std::make_unique<MaskedTable>(N_, iw);
                std::array<MaskedTable *, 1> one{initial_.get()};
                time_sort_ = sorts<std::uint64_t>(one, s, p, {"pdc.time"}, 64);
                if (c.pattern == "P4") {
                    const std::array<unsigned, 9> gw{96, 64, 96, 96, 96, 0, 0, 0, 96};
                    group_input_ = std::make_unique<MaskedTable>(N_, gw);
                    one[0] = group_input_.get();
                    group_sort_ = sorts<uint128_t>(one, s, p, {"pdc.group"}, 96);
                    adjacent_step_ = std::make_unique<EqFresh<std::uint64_t>>(
                        s,
                        p,
                        named_comparison(p, "pdc.group.eq"),
                        difference_.span(),
                        adjacent_.words());
                    boundary_step_ = std::make_unique<FusedAnd2A<1, uint128_t>>(
                        s,
                        p,
                        named_gate(p, "pdc.group.boundary"),
                        std::array<std::span<const std::uint64_t>, 1>{boundary_.words()},
                        group_.span());
                    eqopen_ = {&adjacent_step_->opening()};
                    eqcmp_ = {&adjacent_step_->comparison()};
                    boundary_steps_ = {boundary_step_.get()};
                    eqor_ = std::make_unique<RoundExecutor>(eqopen_);
                    eqcr_ = std::make_unique<RoundExecutor>(eqcmp_);
                    boundary_round_ = std::make_unique<RoundExecutor>(boundary_steps_);
                }
                const auto widths = amount_widths(h_, money_, c.pattern == "P2");
                std::array<MaskedTable *, 2> tables{};
                for (unsigned r = 0; r < 2; ++r) {
                    money_input_[r] = std::make_unique<MaskedTable>(N_, widths);
                    tables[r] = money_input_[r].get();
                }
                money_sort_ =
                    sorts<uint128_t>(tables, s, p, {"pdc.amount.0", "pdc.amount.1"}, money_);
                std::array<MaskedTable *, 2> source{&money_sort_->output(0),
                                                    &money_sort_->output(1)};
                std::array<DegreeLevel *, 2> previous{};
                levels_.reserve(h_);
                for (unsigned it = 0; it < h_; ++it) {
                    const auto ell = h_ - 1 - it;
                    levels_.emplace_back();
                    auto &level = levels_.back();
                    for (unsigned r = 0; r < 2; ++r) {
                        level.runs[r] = std::make_unique<DegreeLevel>(
                            ell, r, h_, c.pattern == "P2", s, p, *source[r], previous[r]);
                        previous[r] = level.runs[r].get();
                        auto &run = *level.runs[r];
                        for (auto &and_gate : run.ands) {
                            level.a.push_back(and_gate.get());
                        }
                        level.a.push_back(run.small_b.get());
                        if (run.wide_b) {
                            level.a.push_back(run.wide_b.get());
                        }
                        level.b.push_back(run.product_step.get());
                        if (ell) {
                            level.b.push_back(run.destination_step.get());
                            level.c.push_back(run.remask.get());
                            level.d.push_back(run.reveal.get());
                            source[r] = run.next.get();
                        }
                    }
                    if (!ell) {
                        level.a.push_back(&post_->omega_step());
                    }
                    level.ar = std::make_unique<RoundExecutor>(level.a);
                    level.br = std::make_unique<RoundExecutor>(level.b);
                    if (ell) {
                        level.cr = std::make_unique<RoundExecutor>(level.c);
                        level.dr = std::make_unique<RoundExecutor>(level.d);
                        std::array<Shuffle *, 2> sh{level.runs[0]->shuffle.get(),
                                                    level.runs[1]->shuffle.get()};
                        level.shuffles = std::make_unique<ShuffleGroup>(sh);
                    }
                }
                if (c.pattern == "P2") {
                    const std::array<unsigned, 2> rw{96, 64};
                    for (unsigned r = 0; r < 2; ++r) {
                        restore_input_[r] = std::make_unique<MaskedTable>(N_, rw);
                        tables[r] = restore_input_[r].get();
                    }
                    restore_sort_ = sorts<std::uint64_t>(
                        tables, s, p, {"pdc.restore.0", "pdc.restore.1"}, std::max(2u, h_ + 1));
                }
            }

            void construct() {
                const auto w = c_.windows[0];
                const auto O = (std::uint64_t{1} << c_.bt) + w.upper + 1;
                for (std::size_t e = 0; e < N_; ++e) {
                    const bool point = e < Kx_, positive = !point && ((e - Kx_) % 2 == 0);
                    const auto row = point ? e : (e - Kx_) / 2;
                    const auto &table = point ? x_ : y_;
                    const auto off = point ? O : O - (positive ? w.lower : w.upper + 1);
                    initial_->columns[0].d64.data()[e] =
                        3 * (table.columns[0].d64.data()[row] + off) + (point ? 1u : 2u);
                    initial_->columns[0].m64.data()[e] = 3 * table.columns[0].m64.data()[row];
                    for (unsigned r = 0; r < 2; ++r) {
                        const auto coef = 3 * (point ? (r ? c_.q + c_.p : c_.q - c_.p) : c_.q);
                        initial_->columns[r + 1].d128.data()[e] =
                            (coef * table.columns[1].d128.data()[row] + (point ? 1 : (r ? 0 : 2))) &
                            native_mask<uint128_t>(96);
                        initial_->columns[r + 1].m128.data()[e] =
                            (coef * table.columns[1].m128.data()[row]) & native_mask<uint128_t>(96);
                    }
                    initial_->columns[3].d128.data()[e] = e;
                    copy_at(table.columns[2],
                            row,
                            initial_->columns[point      ? 4
                                              : positive ? 5
                                                         : 6],
                            e);
                    if (c_.pattern == "P4") {
                        copy_at(table.columns[3], row, initial_->columns[7], e);
                    }
                }
            }

            void group(Net &net) {
                for (std::size_t i = 0; i < N_; ++i) {
                    for (unsigned j = 0; j < 8; ++j) {
                        copy_at(initial_->columns[j], i, group_input_->columns[j + 1], i);
                    }
                    const auto cp = share<uint128_t>(initial_->columns[7], i, party_);
                    const auto id = share<uint128_t>(initial_->columns[3], i, party_);
                    assign_share(group_input_->columns[0],
                                 i,
                                 (cp << (h_ + 1)) + id +
                                     (party_ == 0 && i < Kx_ ? (uint128_t{1} << h_) : 0));
                }
                group_input_->mark_public_reordered();
                group_sort_->run(net);
                record_sort(*group_sort_);
                stats_.counterparty_sort = group_sort_->sort_stats[0];
                auto &sorted = group_sort_->output();
                for (std::size_t i = 1; i < N_; ++i) {
                    difference_.data()[i - 1] =
                        share<std::uint64_t>(sorted.columns[8], i, party_) -
                        share<std::uint64_t>(sorted.columns[8], i - 1, party_);
                }
                eqor_->run(net);
                eqcr_->run(net);
                boundary_.set(0, party_ == 0);
                for (std::size_t i = 1; i < N_; ++i) {
                    boundary_.set(i, adjacent_.get(i - 1) ^ (party_ == 0));
                }
                boundary_round_->run(net);
                uint128_t prefix = 0;
                for (std::size_t i = 0; i < N_; ++i) {
                    prefix += group_.data()[i];
                    group_.data()[i] = prefix - (party_ == 0 ? 1u : 0u);
                    for (unsigned j = 0; j < 8; ++j) {
                        copy_at(sorted.columns[j + 1], i, initial_->columns[j], i);
                    }
                    const auto tau = share<std::uint64_t>(initial_->columns[0], i, party_) +
                                     (static_cast<std::uint64_t>(group_.data()[i]) << (c_.bt + 4));
                    assign_share(initial_->columns[0], i, tau);
                }
                initial_->mark_public_reordered();
            }

            void fold_time() {
                for (std::size_t i = 0; i < N_; ++i) {
                    assign_share(initial_->columns[0],
                                 i,
                                 (share<std::uint64_t>(initial_->columns[0], i, party_) << h_) +
                                     share<std::uint64_t>(initial_->columns[3], i, party_));
                }
                initial_->mark_public_reordered();
            }

            void amounts() {
                auto &time = time_sort_->output();
                for (unsigned r = 0; r < 2; ++r) {
                    for (std::size_t i = 0; i < N_; ++i) {
                        auto &table = *money_input_[r];
                        assign_share(table.columns[0],
                                     i,
                                     (share<uint128_t>(time.columns[r + 1], i, party_) << h_) +
                                         share<uint128_t>(time.columns[3], i, party_));
                        copy_at(time.columns[3], i, table.columns[1], i);
                        for (unsigned j = 0; j < 3; ++j) {
                            copy_at(time.columns[4 + j], i, table.columns[2 + j], i);
                        }
                        for (unsigned bit = 0; bit < h_; ++bit) {
                            table.columns[5 + bit].bits.set(i, (i >> bit) & 1);
                        }
                    }
                }
                for (auto &t : money_input_) {
                    t->mark_public_reordered();
                }
            }

            void one_dimension(Net &net) {
                boolean_share(x_.columns[2], valid_, party_);
                for (std::size_t i = 0; i < Kx_; ++i) {
                    amount_delta_.data()[i] =
                        static_cast<std::uint64_t>(x_.columns[1].d128.data()[i]);
                }
                ar_->run(net);
                br_->run(net);
                const auto O = (std::uint64_t{1} << c_.bt) + c_.windows[0].upper + 1;
                for (std::size_t e = 0; e < N_; ++e) {
                    const bool point = e < Kx_, positive = !point && ((e - Kx_) % 2 == 0);
                    const auto row = point ? e : (e - Kx_) / 2;
                    const auto off = point || positive ? O : O - c_.windows[0].upper - 1;
                    const auto time = 3 * (share<std::uint64_t>(x_.columns[0], row, party_) +
                                           (party_ == 0 ? off : 0)) +
                                      (party_ == 0 ? (point ? 1u : 2u) : 0u);
                    assign_share(initial_->columns[0], e, (time << h_) + (party_ == 0 ? e : 0));
                    initial_->columns[1].d128.data()[e] = e;
                    assign_share(initial_->columns[2], e, point ? weight_.data()[row] : 0ull);
                }
                initial_->mark_public_reordered();
                time_sort_->run(net);
                record_sort(*time_sort_);
                stats_.sorts[0] = time_sort_->sort_stats[0];
                stats_.time_sort_seconds = time_sort_->sort_seconds;
                auto &sorted = time_sort_->output();
                std::uint64_t prefix = 0;
                for (std::size_t i = 0; i < N_; ++i) {
                    copy_at(sorted.columns[1], i, restore_input_[0]->columns[0], i);
                    assign_share(restore_input_[0]->columns[1], i, prefix);
                    prefix += share<std::uint64_t>(sorted.columns[2], i, party_);
                }
                restore_input_[0]->mark_public_reordered();
                restore_sort_->run(net);
                record_sort(*restore_sort_);
                stats_.restore_sorts[0] = restore_sort_->sort_stats[0];
                const auto &restored = restore_sort_->output();
                for (std::size_t y = 0; y < Ky_; ++y) {
                    post_->degrees.data()[y] =
                        share<std::uint64_t>(restored.columns[1], Kx_ + 2 * y, party_) -
                        share<std::uint64_t>(restored.columns[1], Kx_ + 2 * y + 1, party_) -
                        weight_.data()[y];
                }
                post_->aggregate(net);
            }

            void two_dimensions(Net &net) {
                construct();
                if (group_sort_) {
                    group(net);
                }
                fold_time();
                time_sort_->run(net);
                record_sort(*time_sort_);
                stats_.sorts[0] = time_sort_->sort_stats[0];
                stats_.time_sort_seconds = time_sort_->sort_seconds;
                amounts();
                money_sort_->run(net);
                record_sort(*money_sort_);
                stats_.sorts[1] = money_sort_->sort_stats[0];
                stats_.sorts[2] = money_sort_->sort_stats[1];
                stats_.amount_sort_seconds = money_sort_->sort_seconds;
                Timer timer;
                std::array<std::uint64_t, 2> acc{};
                for (auto &level : levels_) {
                    for (auto &r : level.runs) {
                        r->prepare();
                    }
                    level.ar->run(net);
                    for (auto &r : level.runs) {
                        r->prefix();
                    }
                    level.br->run(net);
                    if (c_.pattern != "P2") {
                        for (unsigned r = 0; r < 2; ++r) {
                            for (auto x : level.runs[r]->product.span()) {
                                acc[r] += x;
                            }
                        }
                    }
                    if (level.shuffles) {
                        for (auto &r : level.runs) {
                            r->destinations();
                        }
                        level.cr->run(net);
                        Timer st;
                        level.shuffles->run(net);
                        stats_.shuffle_seconds += st.seconds();
                        stats_.shuffle_rounds += c_.n;
                        level.dr->run(net);
                        for (auto &r : level.runs) {
                            r->scatter();
                        }
                    }
                }
                stats_.levels_seconds = timer.seconds();
                stats_.levels = h_;
                if (c_.pattern != "P2") {
                    post_->count()[0] = acc[0] - acc[1];
                } else {
                    for (unsigned r = 0; r < 2; ++r) {
                        auto &last = *levels_.back().runs[r];
                        for (std::size_t i = 0; i < N_; ++i) {
                            copy_at(last.source.columns[last.source_id()],
                                    i,
                                    restore_input_[r]->columns[0],
                                    i);
                            assign_share(restore_input_[r]->columns[1], i, last.accumulated(i));
                        }
                        restore_input_[r]->mark_public_reordered();
                    }
                    restore_sort_->run(net);
                    record_sort(*restore_sort_);
                    for (unsigned r = 0; r < 2; ++r) {
                        stats_.restore_sorts[r] = restore_sort_->sort_stats[r];
                    }
                    for (std::size_t y = 0; y < Ky_; ++y) {
                        const auto pos = Kx_ + 2 * y, neg = pos + 1;
                        post_->degrees.data()[y] =
                            share<std::uint64_t>(restore_sort_->output(0).columns[1], pos, party_) -
                            share<std::uint64_t>(restore_sort_->output(0).columns[1], neg, party_) -
                            share<std::uint64_t>(restore_sort_->output(1).columns[1], pos, party_) +
                            share<std::uint64_t>(restore_sort_->output(1).columns[1], neg, party_);
                    }
                    post_->aggregate(net);
                }
            }

            void run(Net &net) override {
                if (used_) {
                    throw std::logic_error("pattern DC reused");
                }
                used_ = true;
                const auto before = net.open_stats();
                Timer timer;
                post_->prepare_omega();
                if (empty_) {
                    empty_round_->run(net);
                    post_->aggregate(net);
                } else if (c_.pattern == "P3") {
                    one_dimension(net);
                } else {
                    two_dimensions(net);
                }
                post_->alarm(net);
                const auto after = net.open_stats();
                stats_.seconds = timer.seconds();
                stats_.rounds = after.rounds - before.rounds + stats_.shuffle_rounds;
                stats_.hops = after.hops - before.hops + stats_.shuffle_rounds;
                stats_.contribution_bits = after.contribution_bits - before.contribution_bits;
            }

            std::span<const std::uint64_t> count_shares() const override {
                return post_->count();
            }

            std::uint64_t overflow_share() const override {
                return post_->overflow();
            }

            std::span<const std::uint64_t> alarm_shares() const override {
                return post_->alarms();
            }

            const DCStats &stats() const override {
                return stats_;
            }
        };
    }

    std::unique_ptr<PatternOperator> make_pattern_nl(const InvestigationConfig &,
                                                     MaterialStore &,
                                                     const MaterialPlan &,
                                                     const MaskedTable &,
                                                     const MaskedTable &,
                                                     std::span<const std::uint64_t>,
                                                     std::span<const std::uint64_t>,
                                                     std::span<const std::uint64_t>);

    std::unique_ptr<PatternOperator> make_pattern_operator(const InvestigationConfig &c,
                                                           MaterialStore &s,
                                                           const MaterialPlan &p,
                                                           const MaskedTable &in,
                                                           const MaskedTable &out,
                                                           std::span<const std::uint64_t> oi,
                                                           std::span<const std::uint64_t> oo,
                                                           std::span<const std::uint64_t> extra) {
        c.validate();
        for (auto [table, count] : std::array<std::pair<const MaskedTable *, std::size_t>, 2>{
                 {{&in, c.kin}, {&out, c.kout}}}) {
            const std::vector<unsigned> expected = c.carry_cp()
                                                       ? std::vector<unsigned>{64, 96, 0, 96}
                                                       : std::vector<unsigned>{64, 96, 0};
            if (table->columns.size() != expected.size() || !table->dealer_knows_positions) {
                throw std::invalid_argument("pattern requires remasked fixed slots");
            }
            for (std::size_t j = 0; j < expected.size(); ++j) {
                if (table->columns[j].width != expected[j] || table->columns[j].count != count ||
                    !table->columns[j].dealer_knows_mask) {
                    throw std::invalid_argument("pattern slot shape or positional mask invariant");
                }
            }
        }
        if (c.algorithm == "nl") {
            return make_pattern_nl(c, s, p, in, out, oi, oo, extra);
        }
        return std::make_unique<PatternDC>(c, s, p, in, out, oi, oo, extra);
    }
}
