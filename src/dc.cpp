#include "bj/ops/dc.hpp"
#include "bj/ops/patterns.hpp"
#include "bj/common/timer.hpp"
#include "bj/ops/partition.hpp"
#include "bj/proto/and2a.hpp"
#include "bj/proto/multiply.hpp"
#include "bj/proto/remask.hpp"
#include "bj/proto/shuffle.hpp"

namespace bj {
    void gather_columns(const MaskedTable &src,
                        MaskedTable &dst,
                        std::span<const unsigned> cols,
                        std::span<const std::size_t> order) {
        if (cols.size() != dst.columns.size()) {
            throw std::invalid_argument("gather column count");
        }
        if (!cols.empty()) {
            const auto rows = src.columns.at(cols[0]).count;
            for (auto index : order) {
                if (index >= rows) {
                    throw std::invalid_argument("gather index outside source");
                }
            }
            for (auto column : cols) {
                if (src.columns.at(column).count != rows) {
                    throw std::invalid_argument("gather source row counts");
                }
            }
        }
        for (std::size_t j = 0; j < cols.size(); ++j) {
            const auto &from = src.columns.at(cols[j]);
            auto &to = dst.columns[j];
            if (from.width != to.width || order.size() != to.count) {
                throw std::invalid_argument("gather column shape");
            }
            if (!from.width) {
#pragma omp parallel for if (to.count >= 16384) schedule(static)
                for (std::size_t w = 0; w < to.bits.words().size(); ++w) {
                    std::uint64_t d = 0, m = 0;
                    for (unsigned lane = 0; lane < 64 && w * 64 + lane < to.count; ++lane) {
                        const auto i = order[w * 64 + lane];
                        d |= std::uint64_t(from.bits.get(i)) << lane;
                        m |= std::uint64_t(from.mask_bits.get(i)) << lane;
                    }
                    to.bits.words()[w] = d;
                    to.mask_bits.words()[w] = m;
                }
            } else {
                auto gather = [&]<class T>() {
                    auto d = to.delta<T>(), m = to.mask<T>();
                    const auto fd = from.delta<T>(), fm = from.mask<T>();
#pragma omp parallel for if (to.count >= 16384) schedule(static)
                    for (std::size_t i = 0; i < to.count; ++i) {
                        d[i] = fd[order[i]];
                        m[i] = fm[order[i]];
                    }
                };
                if (from.width <= 32) {
                    gather.template operator()<std::uint32_t>();
                } else if (from.width <= 64) {
                    gather.template operator()<std::uint64_t>();
                } else {
                    gather.template operator()<uint128_t>();
                }
            }
        }
        dst.mark_public_reordered();
    }

    namespace {
        std::size_t shuffle_index(const MaterialPlan &p, const std::string &name) {
            for (std::size_t i = 0; i < p.shuffles.size(); ++i) {
                if (p.shuffles[i].name == name) {
                    return i;
                }
            }
            throw std::invalid_argument("unknown shuffle " + name);
        }

        std::size_t multiply_index(const MaterialPlan &p, const std::string &name) {
            for (std::size_t i = 0; i < p.multiplications.size(); ++i) {
                if (p.multiplications[i].name == name) {
                    return i;
                }
            }
            throw std::invalid_argument("unknown multiply " + name);
        }

        std::vector<std::size_t>
        mask_ids(const MaterialPlan &p, const std::string &prefix, std::size_t columns) {
            std::vector<std::size_t> ids;
            for (std::size_t j = 0; j < columns; ++j) {
                ids.push_back(named_mask(p, prefix + std::to_string(j)));
            }
            return ids;
        }

        std::vector<unsigned> amount_widths(unsigned h) {
            std::vector<unsigned> w(5 + h, 0);
            w[0] = w[1] = 72;
            return w;
        }

        std::vector<unsigned> level_widths(unsigned ell) {
            std::vector<unsigned> w(4 + ell, 0);
            w[0] = 32;
            return w;
        }

        void boolean_share(const MaskedColumn &c, BitVec &out, unsigned party) {
            for (std::size_t w = 0; w < out.words().size(); ++w) {
                out.words()[w] = c.mask_bits.words()[w] ^ (party == 0 ? c.bits.words()[w] : 0);
            }
        }

        struct LevelRun {
            unsigned ell, party, bit_column, marker_column;
            std::size_t N;
            MaskedTable &source;
            LevelRun *previous;
            BitVec pt, ps, ng, b, nb;
            AlignedBuffer<std::uint64_t> u, positive, negative, v, S, product;
            AlignedBuffer<std::uint32_t> ba, Z, factor, dst_product;
            std::array<std::unique_ptr<FusedAnd2A<2, std::uint64_t>>, 3> ands;
            std::unique_ptr<FusedAnd2A<1, std::uint32_t>> b2a;
            std::unique_ptr<BeaverProduct<std::uint64_t>> mul64;
            std::unique_ptr<BeaverProduct<std::uint32_t>> mul32;
            std::unique_ptr<MaskedTable> work, next;
            std::unique_ptr<RemaskStep> remask;
            std::unique_ptr<Shuffle> shuffle;
            std::unique_ptr<RevealDestination> reveal;
            AlignedBuffer<std::size_t> inverse, identity;
            AlignedBuffer<unsigned char> seen;
            std::vector<unsigned> copy_columns;

            LevelRun(unsigned level,
                     unsigned run,
                     unsigned h,
                     MaterialStore &store,
                     const MaterialPlan &plan,
                     MaskedTable &src,
                     std::size_t count,
                     LevelRun *prior)
                : ell(level), party(store.party()),
                  bit_column(level == h - 1 ? 5 + level : 4 + level),
                  marker_column(level == h - 1 ? 2u : 1u), N(count), source(src), previous(prior),
                  pt(N), ps(N), ng(N), b(N), nb(N), u(N), positive(N), negative(N), v(N), S(N),
                  product(N), ba(N), Z(N), factor(N), dst_product(N), inverse(level ? N : 0),
                  identity(level ? N : 0), seen(level ? N : 0) {
                const auto prefix = "dc.level." + std::to_string(level) + "." + std::to_string(run);
                const std::array<std::array<std::span<const std::uint64_t>, 2>, 3> inputs{
                    {{pt.words(), nb.words()}, {ps.words(), b.words()}, {ng.words(), b.words()}}};
                const std::array<std::span<std::uint64_t>, 3> outputs{
                    u.span(), positive.span(), negative.span()};
                for (unsigned j = 0; j < 3; ++j) {
                    ands[j] = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
                        store,
                        plan,
                        named_gate(plan, prefix + ".and." + std::to_string(j)),
                        inputs[j],
                        outputs[j]);
                }
                b2a = std::make_unique<FusedAnd2A<1, std::uint32_t>>(
                    store,
                    plan,
                    named_gate(plan, prefix + ".b2a"),
                    std::array<std::span<const std::uint64_t>, 1>{b.words()},
                    ba.span());
                mul64 = std::make_unique<BeaverProduct<std::uint64_t>>(
                    store,
                    plan,
                    multiply_index(plan, prefix + ".mul64"),
                    v.span(),
                    S.span(),
                    product.span());
                if (level) {
                    mul32 = std::make_unique<BeaverProduct<std::uint32_t>>(
                        store,
                        plan,
                        multiply_index(plan, prefix + ".mul32"),
                        ba.span(),
                        factor.span(),
                        dst_product.span());
                    auto widths = level_widths(level);
                    work = std::make_unique<MaskedTable>(previous ? 0 : N, widths);
                    next = std::make_unique<MaskedTable>(previous ? 0 : N, widths);
                    if (previous) {
                        for (auto &col : work->columns) {
                            col.count = N;
                        }
                        for (auto &col : next->columns) {
                            col.count = N;
                        }
                    }
                    auto ids = mask_ids(plan, prefix + ".mask.", widths.size());
                    remask = std::make_unique<RemaskStep>(*work, store, plan, ids);
                    shuffle =
                        std::make_unique<Shuffle>(*work, store, plan, shuffle_index(plan, prefix));
                    reveal = std::make_unique<RevealDestination>(work->columns[0], party);
                    std::iota(identity.span().begin(), identity.span().end(), 0);
                    copy_columns.push_back(0);
                    for (unsigned j = 0; j < 3 + level; ++j) {
                        copy_columns.push_back(marker_column + j);
                    }
                }
            }

            void prepare() {
                boolean_share(source.columns[marker_column], pt, party);
                boolean_share(source.columns[marker_column + 1], ps, party);
                boolean_share(source.columns[marker_column + 2], ng, party);
                boolean_share(source.columns[bit_column], b, party);
                for (std::size_t w = 0; w < nb.words().size(); ++w) {
                    nb.words()[w] = b.words()[w] ^ (party == 0 ? UINT64_MAX : 0);
                }
                if (N % 64 && !nb.words().empty()) {
                    nb.words().back() &= low_mask64(static_cast<unsigned>(N % 64));
                }
            }

            void prefixes() {
                const std::size_t half = std::size_t{1} << ell, block = half * 2;
#pragma omp parallel for if (N >= 16384) schedule(static)
                for (std::size_t blockno = 0; blockno < (N + block - 1) / block; ++blockno) {
                    const auto start = blockno * block, end = std::min(N, start + block),
                               zero_count = std::min(half, end - start);
                    std::uint64_t sum = 0;
                    std::uint32_t zeros = 0;
                    for (std::size_t i = start; i < end; ++i) {
                        v.data()[i] = positive.data()[i] - negative.data()[i];
                        S.data()[i] = sum;
                        sum += u.data()[i];
                        Z.data()[i] = zeros;
                        if (ell) {
                            factor.data()[i] =
                                (party == 0 ? static_cast<std::uint32_t>(zero_count + i - start)
                                            : 0u) -
                                2 * zeros;
                        }
                        zeros += (party == 0 ? 1u : 0u) - ba.data()[i];
                    }
                }
            }

            std::uint64_t accumulate() {
                std::uint64_t result = 0;
#pragma omp parallel for reduction(+ : result) if (N >= 16384) schedule(static)
                for (std::size_t i = 0; i < N; ++i) {
                    result += product.data()[i];
                }
                return result;
            }

            void destinations() {
                if (!ell) {
                    return;
                }
                if (previous) {
                    for (std::size_t j = 0; j < work->columns.size(); ++j) {
                        work->columns[j] = std::move(previous->work->columns[j]);
                    }
                }
                auto &dst = work->columns[0];
                const auto block = std::size_t{1} << (ell + 1);
                for (std::size_t i = 0; i < N; ++i) {
                    dst.d32.data()[i] = 0;
                    dst.m32.data()[i] =
                        0u - ((party == 0 ? static_cast<std::uint32_t>((i / block) * block) : 0u) +
                              Z.data()[i] + dst_product.data()[i]);
                }
                for (std::size_t j = 1; j < work->columns.size(); ++j) {
                    auto &to = work->columns[j];
                    const auto &from = source.columns[copy_columns[j]];
                    std::copy(from.bits.words().begin(),
                              from.bits.words().end(),
                              to.bits.words().begin());
                    std::copy(from.mask_bits.words().begin(),
                              from.mask_bits.words().end(),
                              to.mask_bits.words().begin());
                }
                work->mark_public_reordered();
                if (previous) {
                    for (std::size_t j = 0; j < next->columns.size(); ++j) {
                        next->columns[j] = std::move(previous->next->columns[j]);
                    }
                }
            }

            void scatter() {
                std::fill(seen.span().begin(), seen.span().end(), 0);
                const auto dst = reveal->positions();
                for (std::size_t i = 0; i < N; ++i) {
                    if (dst[i] >= N || seen.data()[dst[i]]++) {
                        throw std::runtime_error("DC opened destination is not a permutation");
                    }
                    inverse.data()[dst[i]] = i;
                }

                for (unsigned j = 0; j < copy_columns.size(); ++j) {
                    copy_columns[j] = j;
                }
                gather_columns(*work, *next, copy_columns, inverse.span());
            }
        };
    }

    struct DominanceCount::Impl {
        InvestigationConfig config;
        unsigned party, h;
        std::size_t N;
        const MaskedTable &in, &out;
        std::span<const std::uint64_t> oi, oo;
        AlignedBuffer<std::uint64_t> counts{1}, nonoverflow{1};
        BitVec negin{1}, negout{1}, alarms{1};
        AlignedBuffer<std::uint32_t> alert_difference{1};
        std::unique_ptr<FusedAnd2A<2, std::uint64_t>> omega;
        std::unique_ptr<MsbFresh<std::uint32_t>> alert;
        std::array<Step *, 1> omega_steps{}, alert_open{}, alert_cmp{};
        std::unique_ptr<RoundExecutor> trivial_round, alert_open_round, alert_cmp_round;
        std::unique_ptr<MaskedTable> initial, time_sorted;
        std::array<std::unique_ptr<MaskedTable>, 2> amounts, amount_sorted;
        std::unique_ptr<Shuffle> time_shuffle;
        std::array<std::unique_ptr<Shuffle>, 2> amount_shuffle;
        std::unique_ptr<ShuffleGroup> amount_shuffles;
        std::array<std::unique_ptr<RemaskStep>, 2> amount_remask;
        std::array<Step *, 2> amount_remask_steps{};
        std::unique_ptr<RoundExecutor> amount_remask_round;
        std::unique_ptr<PublicQuickSort<std::uint64_t>> time_sort;
        std::array<std::unique_ptr<PublicQuickSort<uint128_t>>, 2> money_sort;
        std::unique_ptr<FreshMsbPool<std::uint64_t>> time_pool;
        std::array<std::unique_ptr<FreshMsbPool<uint128_t>>, 2> money_pool;
        std::unique_ptr<SortGroup<std::uint64_t>> time_group;
        std::unique_ptr<SortGroup<uint128_t>> money_group;

        struct Level {
            std::array<std::unique_ptr<LevelRun>, 2> runs;
            std::vector<Step *> a, b, c, d;
            std::unique_ptr<RoundExecutor> ar, br, cr, dr;
            std::unique_ptr<ShuffleGroup> shuffles;
        };

        std::vector<Level> levels;
        std::vector<unsigned> time_columns, amount_columns;
        DCStats stats;
        bool used = false;

        Impl(const InvestigationConfig &c,
             MaterialStore &s,
             const MaterialPlan &p,
             const MaskedTable &input,
             const MaskedTable &output,
             std::span<const std::uint64_t> i,
             std::span<const std::uint64_t> o)
            : config(c), party(s.party()), h(static_cast<unsigned>(std::bit_width(
                                               c.kin + 2 * c.kout ? c.kin + 2 * c.kout - 1 : 0))),
              N(c.kin + 2 * c.kout), in(input), out(output), oi(i), oo(o) {
            std::fill(counts.span().begin(), counts.span().end(), 0);
            c.validate();
            if (c.algorithm != "dc" || i.size() != 1 || o.size() != 1 || in.columns.size() != 3 ||
                out.columns.size() != 3 || in.columns[0].count != c.kin ||
                out.columns[0].count != c.kout) {
                throw std::invalid_argument("DC slot shape/operator");
            }
            omega = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
                s,
                p,
                named_gate(p, "output.omega"),
                std::array<std::span<const std::uint64_t>, 2>{negin.words(), negout.words()},
                nonoverflow.span());
            omega_steps[0] = omega.get();
            if (c.output == "alarm") {

                if (c.kin && c.kout) {
                    alert =
                        std::make_unique<MsbFresh<std::uint32_t>>(s,
                                                                  p,
                                                                  named_comparison(p, "dc.alarm"),
                                                                  alert_difference.span(),
                                                                  alarms.words());
                    alert_open[0] = &alert->opening();
                    alert_cmp[0] = &alert->comparison();
                    alert_open_round = std::make_unique<RoundExecutor>(alert_open);
                    alert_cmp_round = std::make_unique<RoundExecutor>(alert_cmp);
                }
            }
            if (!c.kin || !c.kout) {
                trivial_round = std::make_unique<RoundExecutor>(omega_steps);
                return;
            }
            const std::array<unsigned, 7> widths{64, 72, 72, 72, 0, 0, 0};
            initial = std::make_unique<MaskedTable>(N, widths);
            time_sorted = std::make_unique<MaskedTable>(N, widths);
            time_columns = {0, 1, 2, 3, 4, 5, 6};
            time_shuffle = std::make_unique<Shuffle>(*initial, s, p, shuffle_index(p, "dc.time"));
            time_sort = std::make_unique<PublicQuickSort<std::uint64_t>>(N);
            time_pool = std::make_unique<FreshMsbPool<std::uint64_t>>(
                s, p, named_comparison(p, "dc.sort.0"), N);
            std::array<PublicQuickSort<std::uint64_t> *, 1> ts{time_sort.get()};
            std::array<FreshMsbPool<std::uint64_t> *, 1> tp{time_pool.get()};
            time_group = std::make_unique<SortGroup<std::uint64_t>>(ts, tp, N, 64);
            const auto aw = amount_widths(h);
            for (unsigned r = 0; r < 2; ++r) {
                amounts[r] = std::make_unique<MaskedTable>(N, aw);
                amount_sorted[r] = std::make_unique<MaskedTable>(N, aw);
                auto ids = mask_ids(p, "dc.amount." + std::to_string(r) + ".mask.", aw.size());
                amount_remask[r] = std::make_unique<RemaskStep>(*amounts[r], s, p, ids);
                amount_remask_steps[r] = amount_remask[r].get();
                amount_shuffle[r] = std::make_unique<Shuffle>(
                    *amounts[r], s, p, shuffle_index(p, "dc.amount." + std::to_string(r)));
                money_sort[r] = std::make_unique<PublicQuickSort<uint128_t>>(N);
                money_pool[r] = std::make_unique<FreshMsbPool<uint128_t>>(
                    s, p, named_comparison(p, "dc.sort." + std::to_string(r + 1)), N);
            }
            amount_remask_round = std::make_unique<RoundExecutor>(amount_remask_steps);
            std::array<Shuffle *, 2> sh{amount_shuffle[0].get(), amount_shuffle[1].get()};
            amount_shuffles = std::make_unique<ShuffleGroup>(sh);
            std::array<PublicQuickSort<uint128_t> *, 2> ms{money_sort[0].get(),
                                                           money_sort[1].get()};
            std::array<FreshMsbPool<uint128_t> *, 2> mp{money_pool[0].get(), money_pool[1].get()};
            money_group = std::make_unique<SortGroup<uint128_t>>(ms, mp, N, 72);
            for (unsigned j = 0; j < aw.size(); ++j) {
                amount_columns.push_back(j);
            }
            levels.reserve(h);
            std::array<MaskedTable *, 2> sources{amount_sorted[0].get(), amount_sorted[1].get()};
            std::array<LevelRun *, 2> previous{};
            for (unsigned it = 0; it < h; ++it) {
                const auto ell = h - 1 - it;
                levels.emplace_back();
                auto &level = levels.back();
                for (unsigned r = 0; r < 2; ++r) {
                    level.runs[r] =
                        std::make_unique<LevelRun>(ell, r, h, s, p, *sources[r], N, previous[r]);
                    previous[r] = level.runs[r].get();
                    auto &run = *level.runs[r];
                    for (auto &gate : run.ands) {
                        level.a.push_back(gate.get());
                    }
                    level.a.push_back(run.b2a.get());
                    level.b.push_back(run.mul64.get());
                    if (ell) {
                        level.b.push_back(run.mul32.get());
                        level.c.push_back(run.remask.get());
                        level.d.push_back(run.reveal.get());
                        sources[r] = run.next.get();
                    }
                }
                if (!ell) {
                    level.a.push_back(omega.get());
                }
                level.ar = std::make_unique<RoundExecutor>(level.a);
                level.br = std::make_unique<RoundExecutor>(level.b);
                if (ell) {
                    level.cr = std::make_unique<RoundExecutor>(level.c);
                    level.dr = std::make_unique<RoundExecutor>(level.d);
                    std::array<Shuffle *, 2> ss{level.runs[0]->shuffle.get(),
                                                level.runs[1]->shuffle.get()};
                    level.shuffles = std::make_unique<ShuffleGroup>(ss);
                }
            }
        }

        void construct() {
            const auto w = config.windows[0];
            const auto O = (std::uint64_t{1} << config.bt) + w.upper + 1;
            const auto reduce = native_mask<uint128_t>(72);
            for (std::size_t e = 0; e < N; ++e) {
                const bool point = e < config.kin;
                const auto row = point ? e : (e - config.kin) % config.kout;
                const auto &table = point ? in : out;
                const auto offset =
                    point ? O : O - (e < config.kin + config.kout ? w.lower : w.upper + 1);
                auto &tau = initial->columns[0];
                tau.d64.data()[e] =
                    (3 * (table.columns[0].d64.data()[row] + offset) + (point ? 1 : 2)) *
                        (std::uint64_t{1} << h) +
                    e;
                tau.m64.data()[e] = (3 * table.columns[0].m64.data()[row]) << h;
                for (unsigned r = 0; r < 2; ++r) {
                    const auto coef =
                        3 * (point ? (r ? config.q + config.p : config.q - config.p) : config.q);
                    auto &a = initial->columns[r + 1];
                    a.d128.data()[e] =
                        ((coef * table.columns[1].d128.data()[row] + (point ? 1 : (r ? 0 : 2))) *
                             (uint128_t{1} << h) +
                         e) &
                        reduce;
                    a.m128.data()[e] = ((coef * table.columns[1].m128.data()[row]) << h) & reduce;
                }
                initial->columns[3].d128.data()[e] = e;
                const auto marker = point ? 4u : (e < config.kin + config.kout ? 5u : 6u);
                initial->columns[marker].bits.set(e, table.columns[2].bits.get(row));
                initial->columns[marker].mask_bits.set(e, table.columns[2].mask_bits.get(row));
            }
        }

        template <class T>
        void keys(const MaskedTable &table,
                  PublicQuickSort<T> &sort,
                  unsigned key_col,
                  unsigned id_col) {
            auto keys = sort.keys();
            const auto &key = table.columns[key_col];
            (void)id_col;
            const auto mask = native_mask<T>(sizeof(T) == 8 ? 64 : 72);
            for (std::size_t e = 0; e < N; ++e) {
                const T kd = key.delta<T>()[e], km = key.mask<T>()[e];
                keys[e] = ((party == 0 ? kd : T{0}) - km) & mask;
            }
        }

        void build_amounts() {
            for (unsigned r = 0; r < 2; ++r) {
                auto &target = *amounts[r];
                const std::array<unsigned, 5> cols{r + 1, 3, 4, 5, 6};
                for (unsigned j = 0; j < 5; ++j) {
                    auto &to = target.columns[j];
                    const auto &from = time_sorted->columns[cols[j]];
                    if (to.width) {
                        std::copy(from.d128.span().begin(),
                                  from.d128.span().end(),
                                  to.d128.span().begin());
                        std::copy(from.m128.span().begin(),
                                  from.m128.span().end(),
                                  to.m128.span().begin());
                    } else {
                        std::copy(from.bits.words().begin(),
                                  from.bits.words().end(),
                                  to.bits.words().begin());
                        std::copy(from.mask_bits.words().begin(),
                                  from.mask_bits.words().end(),
                                  to.mask_bits.words().begin());
                    }
                }
                for (unsigned bit = 0; bit < h; ++bit) {
                    for (std::size_t i = 0; i < N; ++i) {
                        target.columns[5 + bit].bits.set(i, (i >> bit) & 1);
                    }
                }
                target.mark_public_reordered();
            }
        }

        void run(Net &net) {
            if (used) {
                throw std::logic_error("DC already executed");
            }
            used = true;
            const auto before = net.open_stats();
            Timer all;
            negin.words()[0] = oi[0] ^ (party == 0 ? 1u : 0u);
            negout.words()[0] = oo[0] ^ (party == 0 ? 1u : 0u);
            if (trivial_round) {
                trivial_round->run(net);
            } else {
                construct();
                Timer ts;
                time_shuffle->run(net);
                stats.shuffle_seconds += ts.seconds();
                stats.shuffle_rounds += config.n;
                keys(*initial, *time_sort, 0, 3);
                ts = Timer();
                time_group->run(net);
                gather_columns(*initial, *time_sorted, time_columns, time_sort->order());
                stats.time_sort_seconds = ts.seconds();
                build_amounts();
                amount_remask_round->run(net);
                ts = Timer();
                amount_shuffles->run(net);
                stats.shuffle_seconds += ts.seconds();
                stats.shuffle_rounds += config.n;
                for (unsigned r = 0; r < 2; ++r) {
                    keys(*amounts[r], *money_sort[r], 0, 1);
                }
                ts = Timer();
                money_group->run(net);
                for (unsigned r = 0; r < 2; ++r) {
                    gather_columns(
                        *amounts[r], *amount_sorted[r], amount_columns, money_sort[r]->order());
                }
                stats.amount_sort_seconds = ts.seconds();
                ts = Timer();
                std::array<std::uint64_t, 2> acc{};
                for (auto &level : levels) {
                    for (auto &r : level.runs) {
                        r->prepare();
                    }
                    level.ar->run(net);
                    for (auto &r : level.runs) {
                        r->prefixes();
                    }
                    level.br->run(net);
                    for (unsigned r = 0; r < 2; ++r) {
                        acc[r] += level.runs[r]->accumulate();
                    }
                    if (level.shuffles) {
                        for (auto &r : level.runs) {
                            r->destinations();
                        }
                        level.cr->run(net);
                        Timer sh;
                        level.shuffles->run(net);
                        stats.shuffle_seconds += sh.seconds();
                        stats.shuffle_rounds += config.n;
                        level.dr->run(net);
                        for (auto &r : level.runs) {
                            r->scatter();
                        }
                    }
                }
                counts.data()[0] = acc[0] - acc[1];
                stats.levels_seconds = ts.seconds();
                stats.sorts = {time_sort->stats(time_pool->capacity()),
                               money_sort[0]->stats(money_pool[0]->capacity()),
                               money_sort[1]->stats(money_pool[1]->capacity())};
                stats.levels = h;
            }
            if (config.output == "alarm") {
                if (alert) {
                    alert_difference.data()[0] =
                        static_cast<std::uint32_t>(counts.data()[0]) -
                        (party == 0 ? static_cast<std::uint32_t>(config.threshold) : 0u);
                    alert_open_round->run(net);
                    alert_cmp_round->run(net);
                    if (party == 0) {
                        alarms.words()[0] ^= 1;
                    }
                } else {
                    alarms.words()[0] = party == 0 && config.threshold == 0;
                }
            }
            const auto after = net.open_stats();
            stats.seconds = all.seconds();
            stats.rounds = after.rounds - before.rounds + stats.shuffle_rounds;
            stats.hops = after.hops - before.hops + stats.shuffle_rounds;
            stats.contribution_bits = after.contribution_bits - before.contribution_bits;
        }
    };

    DominanceCount::DominanceCount(const InvestigationConfig &c,
                                   MaterialStore &s,
                                   const MaterialPlan &p,
                                   const MaskedTable &i,
                                   const MaskedTable &o,
                                   std::span<const std::uint64_t> oi,
                                   std::span<const std::uint64_t> oo)
        : impl_(c.pattern == "P1" ? std::make_unique<Impl>(c, s, p, i, o, oi, oo) : nullptr) {
        if (c.pattern != "P1") {
            pattern_ = make_pattern_operator(c, s, p, i, o, oi, oo);
        }
    }

    DominanceCount::~DominanceCount() = default;

    void DominanceCount::run(Net &n) {
        if (pattern_) {
            pattern_->run(n);
        } else {
            impl_->run(n);
        }
    }

    std::span<const std::uint64_t> DominanceCount::count_shares() const {
        return pattern_ ? pattern_->count_shares() : impl_->counts.span();
    }

    std::uint64_t DominanceCount::overflow_share() const {
        return pattern_ ? pattern_->overflow_share()
                        : (impl_->party == 0 ? 1u : 0u) - impl_->nonoverflow.data()[0];
    }

    std::span<const std::uint64_t> DominanceCount::alarm_shares() const {
        return pattern_ ? pattern_->alarm_shares() : impl_->alarms.words();
    }

    const DCStats &DominanceCount::stats() const {
        return pattern_ ? pattern_->stats() : impl_->stats;
    }
}
