#include "bj/ops/masked_sort.hpp"
#include "bj/ops/patterns.hpp"
#include "bj/plan/multi.hpp"
#include <cstring>
#include <nlohmann/json.hpp>

namespace bj {
    namespace {
        void copy_column(const MaskedColumn &a, MaskedColumn &b) {
            if (a.width != b.width || a.count != b.count) {
                throw std::invalid_argument("multi column mismatch");
            }
            if (!a.width) {
                std::copy(a.bits.words().begin(), a.bits.words().end(), b.bits.words().begin());
                std::copy(a.mask_bits.words().begin(),
                          a.mask_bits.words().end(),
                          b.mask_bits.words().begin());
            } else {
                auto copy = [&]<class T>() {
                    std::copy(a.delta<T>().begin(), a.delta<T>().end(), b.delta<T>().begin());
                    std::copy(a.mask<T>().begin(), a.mask<T>().end(), b.mask<T>().begin());
                };
                if (a.width <= 32) {
                    copy.template operator()<std::uint32_t>();
                } else if (a.width <= 64) {
                    copy.template operator()<std::uint64_t>();
                } else {
                    copy.template operator()<uint128_t>();
                }
            }
            b.dealer_knows_mask = a.dealer_knows_mask;
        }

        void bit_share(const MaskedColumn &c, BitVec &out, unsigned party) {
            for (std::size_t w = 0; w < out.words().size(); ++w) {
                out.words()[w] = c.mask_bits.words()[w] ^ (party == 0 ? c.bits.words()[w] : 0);
            }
        }

        struct UnionNode {
            BitVec left, right, result;
            AlignedBuffer<std::uint32_t> product;
            BitVec *a;
            BitVec *b;
            std::unique_ptr<FusedAnd2A<2, std::uint32_t>> step;

            UnionNode(std::size_t n, BitVec *x, BitVec *y)
                : left(n), right(n), result(n), product(n), a(x), b(y) {
            }
        };

        struct RowUnion {
            unsigned party;
            const InvestigationConfig &c;
            const InputManager &input;
            const MaterialPlan &plan;
            std::vector<BitVec> leaves;
            std::vector<std::vector<std::size_t>> input_ids;
            BitVec identity;
            BitVec *result = nullptr;
            bool all = false;
            std::vector<std::vector<std::unique_ptr<UnionNode>>> nodes;
            std::vector<std::vector<Step *>> steps;
            std::vector<std::unique_ptr<RoundExecutor>> rounds;

            RowUnion(const Plan &config,
                     MaterialStore &s,
                     const MaterialPlan &p,
                     const InputManager &in)
                : party(s.party()), c(config.common), input(in), plan(p), identity(c.size()) {
                all = std::any_of(config.templates.begin(), config.templates.end(), [](auto &q) {
                    return q.filter.empty();
                });
                if (c.filters.empty() || all) {
                    result = &identity;
                    return;
                }
                for (const auto &f : c.filters) {
                    input_ids.emplace_back();
                    for (unsigned bank = 0; bank < c.n; ++bank) {
                        input_ids.back().push_back(
                            named_mask(p, "f_" + f + "_" + std::to_string(bank)));
                    }
                }
                leaves.reserve(c.filters.size());
                for (const auto &f : c.filters) {
                    (void)f;
                    leaves.emplace_back(c.size());
                }
                std::vector<BitVec *> prev;
                for (auto &leaf : leaves) {
                    prev.push_back(&leaf);
                }
                for (std::size_t depth = 0; prev.size() > 1; ++depth) {
                    nodes.emplace_back();
                    steps.emplace_back();
                    std::vector<BitVec *> next;
                    for (std::size_t i = 0; i + 1 < prev.size(); i += 2) {
                        auto node = std::make_unique<UnionNode>(c.size(), prev[i], prev[i + 1]);
                        node->step = std::make_unique<FusedAnd2A<2, std::uint32_t>>(
                            s,
                            p,
                            named_gate(p,
                                       "multi.union." + std::to_string(depth) + "." +
                                           std::to_string(i / 2)),
                            std::array<std::span<const std::uint64_t>, 2>{node->left.words(),
                                                                          node->right.words()},
                            node->product.span());
                        next.push_back(&node->result);
                        steps.back().push_back(node->step.get());
                        nodes.back().push_back(std::move(node));
                    }
                    if (prev.size() % 2) {
                        next.push_back(prev.back());
                    }
                    rounds.push_back(std::make_unique<RoundExecutor>(steps.back()));
                    prev = std::move(next);
                }
                result = prev[0];
            }

            void run(Net &net) {
                if (c.filters.empty() || all) {
                    if (party == 0) {
                        for (std::size_t i = 0; i < identity.size(); ++i) {
                            identity.set(i, true);
                        }
                    }
                    return;
                }
                for (std::size_t f = 0; f < c.filters.size(); ++f) {
                    std::size_t off = 0;
                    for (unsigned bank = 0; bank < c.n; ++bank) {
                        const auto &value = input.at(input_ids[f][bank]);
                        for (std::size_t i = 0; i < c.rows[bank]; ++i) {
                            leaves[f].set(off + i, value.share_bits.get(i));
                        }
                        off += c.rows[bank];
                    }
                }
                for (std::size_t d = 0; d < nodes.size(); ++d) {
                    for (auto &n : nodes[d]) {
                        for (std::size_t i = 0; i < c.size(); ++i) {
                            n->left.set(i, n->a->get(i) ^ (party == 0));
                            n->right.set(i, n->b->get(i) ^ (party == 0));
                        }
                    }
                    rounds[d]->run(net);
                    for (auto &n : nodes[d]) {
                        for (std::size_t i = 0; i < c.size(); ++i) {
                            n->result.set(i, (n->product.data()[i] & 1) ^ (party == 0));
                        }
                    }
                }
            }
        };

        struct TemplateRun {
            double elapsed = 0;
            const QueryTemplate &q;
            unsigned party;
            const MaskedTable &source_in;
            const MaskedTable &source_out;
            MaskedTable in, out;
            MaterialPlan view;
            const MaskedColumn *fi = nullptr;
            const MaskedColumn *fo = nullptr;
            BitVec x, y, extra;
            AlignedBuffer<std::uint32_t> product, vin, vout;
            std::vector<std::unique_ptr<Step>> filter_steps;
            std::vector<Step *> filter_ptr, remask_ptr;
            std::unique_ptr<RoundExecutor> filter_round, remask_round;
            std::unique_ptr<RemaskStep> remask_in, remask_out;
            std::unique_ptr<NestedLoop> nl;
            std::unique_ptr<PatternOperator> op;

            TemplateRun(const QueryTemplate &query,
                        const InvestigationConfig &common,
                        MaterialStore &s,
                        const MaterialPlan &p,
                        const Compact &compact,
                        std::size_t index,
                        bool align_nl)
                : q(query), party(s.party()), source_in(compact.inbound()),
                  source_out(compact.outbound()),
                  in(q.config.kin,
                     q.config.carry_cp() ? std::vector<unsigned>{64, 96, 0, 96}
                                         : std::vector<unsigned>{64, 96, 0}),
                  out(q.config.kout, table_widths_of(in)), view(template_material_view(p, index)),
                  x(q.config.algorithm == "nl" ? q.config.pairs() : q.config.kin),
                  y(q.config.algorithm == "nl" ? q.config.pairs() : q.config.kout),
                  extra(!q.filter.empty() ? (q.config.algorithm == "nl" ? q.config.pairs()
                                             : q.config.pattern == "P2" ? q.config.kout
                                                                        : 0)
                                          : 0),
                  product(extra.size()), vin(q.config.kin), vout(q.config.kout) {
                if (!q.filter.empty()) {
                    const auto f = static_cast<std::size_t>(
                        std::find(common.filters.begin(), common.filters.end(), q.filter) -
                        common.filters.begin());
                    fi = &source_in.columns[3 + (common.carry_cp() ? 1 : 0) + f];
                    fo = &source_out.columns[3 + (common.carry_cp() ? 1 : 0) + f];
                    const auto pre = "q" + std::to_string(index) + ".";
                    if (q.config.algorithm == "nl") {
                        filter_steps.push_back(std::make_unique<FusedAnd2A<2, std::uint32_t>>(
                            s,
                            p,
                            named_gate(p, pre + "extra.pair"),
                            std::array<std::span<const std::uint64_t>, 2>{x.words(), y.words()},
                            product.span()));
                    } else {

                        for (unsigned side = 0; side < 2; ++side) {
                            auto &table = side ? out : in;
                            auto bits = side ? y.words() : x.words();
                            filter_steps.push_back(std::make_unique<FusedAnd2A<2, std::uint32_t>>(
                                s,
                                p,
                                named_gate(p, pre + "extra." + (side ? "out" : "in")),
                                std::array<std::span<const std::uint64_t>, 2>{
                                    bits, table.columns[2].mask_bits.words()},
                                side ? vout.span() : vin.span()));
                        }
                        const std::array<unsigned, 1> col{2};
                        std::array<std::size_t, 1> im{named_mask(p, pre + "slot.vin")},
                            om{named_mask(p, pre + "slot.vout")};
                        remask_in = std::make_unique<RemaskStep>(in, s, p, im, col);
                        remask_out = std::make_unique<RemaskStep>(out, s, p, om, col);
                        remask_ptr = {remask_in.get(), remask_out.get()};
                        remask_round = std::make_unique<RoundExecutor>(remask_ptr);
                    }
                    for (auto &step : filter_steps) {
                        filter_ptr.push_back(step.get());
                    }
                    filter_round = std::make_unique<RoundExecutor>(filter_ptr);
                }
                if (!filter_round && align_nl && q.config.algorithm == "nl") {
                    filter_round = std::make_unique<RoundExecutor>(filter_ptr);
                }
                if (q.config.algorithm == "nl") {
                    nl = std::make_unique<NestedLoop>(q.config,
                                                      s,
                                                      view,
                                                      in,
                                                      out,
                                                      compact.omega_in(),
                                                      compact.omega_out(),
                                                      "output.omega",
                                                      extra.words());
                } else {
                    op = make_pattern_operator(q.config,
                                               s,
                                               view,
                                               in,
                                               out,
                                               compact.omega_in(),
                                               compact.omega_out(),
                                               extra.words());
                }
            }

            void prepare_filter(Net &net) {
                for (std::size_t j = 0; j < in.columns.size(); ++j) {
                    copy_column(source_in.columns[j], in.columns[j]);
                    copy_column(source_out.columns[j], out.columns[j]);
                }
                if (!fi) {
                    if (filter_round) {
                        filter_round->run(net);
                    }
                    return;
                }
                if (q.config.algorithm == "nl") {
                    const auto &fx = q.config.pattern == "P4" ? *fo : *fi;
                    const auto &fy =
                        q.config.pattern == "P2" || q.config.pattern == "P1" ? *fo : *fi;
                    const auto K = q.config.y_size();
                    for (std::size_t i = 0; i < q.config.pairs(); ++i) {
                        const auto a = i / K, b = i % K;
                        x.set(i, fx.mask_bits.get(a) ^ (party == 0 ? fx.bits.get(a) : 0));
                        y.set(i, fy.mask_bits.get(b) ^ (party == 0 ? fy.bits.get(b) : 0));
                    }
                    filter_round->run(net);
                    for (std::size_t i = 0; i < extra.size(); ++i) {
                        extra.set(i, product.data()[i] & 1);
                    }
                } else {
                    bit_share(*fi, x, party);
                    bit_share(*fo, y, party);
                    if (q.config.pattern == "P2") {
                        bit_share(*fo, extra, party);
                    }
                    for (auto *table : {&in, &out}) {
                        auto &v = table->columns[2];
                        for (std::size_t w = 0; w < v.mask_bits.words().size(); ++w) {
                            v.mask_bits.words()[w] ^= party == 0 ? v.bits.words()[w] : 0;
                        }
                    }
                    filter_round->run(net);
                    for (unsigned side = 0; side < 2; ++side) {
                        auto &v = (side ? out : in).columns[2];
                        auto values = side ? vout.span() : vin.span();
                        for (std::size_t i = 0; i < v.count; ++i) {
                            v.bits.set(i, false);
                            v.mask_bits.set(i, values[i] & 1);
                        }
                        v.dealer_knows_mask = false;
                    }
                    remask_round->run(net);
                }
            }

            void run(Net &net) {
                Timer timer;
                prepare_filter(net);
                if (nl) {
                    nl->run(net);
                } else {
                    op->run(net);
                }
                elapsed = timer.seconds();
            }

            std::span<const std::uint64_t> counts() const {
                return nl ? nl->count_shares() : op->count_shares();
            }

            std::span<const std::uint64_t> alarms() const {
                return nl ? nl->alarm_shares() : op->alarm_shares();
            }

            std::uint64_t overflow() const {
                return nl ? nl->overflow_share() : op->overflow_share();
            }
        };

        struct ThresholdGroup {
            std::vector<SharedIntervalHandle> handles;
            std::vector<const void *> inputs;
            std::vector<unsigned> native_bytes;
            std::vector<bool> registered;
            std::vector<std::span<std::uint64_t>> outputs;
            std::unique_ptr<Interval<std::uint32_t>> i32;
            std::unique_ptr<Interval<std::uint64_t>> i64;
            std::vector<Step *> steps;
            std::unique_ptr<RoundExecutor> round;
            std::size_t count = 0, root_position = 0;
            bool done = false;

            explicit ThresholdGroup(std::size_t n)
                : handles(n), inputs(n), native_bytes(n), registered(n, false), outputs(n) {
            }

            void prepare(MaterialStore &s,
                         const MaterialPlan &p,
                         std::size_t root,
                         std::span<const BandWindow> windows,
                         std::size_t position) {
                root_position = position;
                count = p.comparisons[root].count;
                if (!std::all_of(registered.begin(),
                                 registered.end(),
                                 [](bool b) {
                                     return b;
                                 }) ||
                    !std::all_of(native_bytes.begin(), native_bytes.end(), [&](auto b) {
                        return b == native_bytes[position];
                    })) {
                    throw std::logic_error("shared endpoint binding mismatch");
                }
                if (native_bytes[position] == 4) {
                    i32 = std::make_unique<Interval<std::uint32_t>>(
                        s,
                        p,
                        root,
                        std::span(static_cast<const std::uint32_t *>(inputs[position]), count),
                        windows,
                        outputs);
                    steps = {i32.get()};
                } else if (native_bytes[position] == 8) {
                    i64 = std::make_unique<Interval<std::uint64_t>>(
                        s,
                        p,
                        root,
                        std::span(static_cast<const std::uint64_t *>(inputs[position]), count),
                        windows,
                        outputs);
                    steps = {i64.get()};
                } else {
                    throw std::invalid_argument("shared time native width");
                }
                round = std::make_unique<RoundExecutor>(steps);
            }

            OpenBuf *request() {
                if (done || std::none_of(handles.begin(), handles.end(), [](auto &h) {
                        return h.requested;
                    })) {
                    return nullptr;
                }
                if (!std::all_of(handles.begin(), handles.end(), [](auto &h) {
                        return h.requested;
                    })) {
                    throw std::logic_error("shared endpoint readiness mismatch");
                }
                if (count) {
                    for (auto ptr : inputs) {
                        if (std::memcmp(
                                ptr, inputs[root_position], count * native_bytes[root_position])) {
                            throw std::logic_error("shared endpoint masked operands differ");
                        }
                    }
                }
                return &round->prepare_open();
            }

            void consume() {
                round->consume_open();
                done = true;
                for (auto &h : handles) {
                    h.done = true;
                }
            }
        };

        struct TimeGroup {
            std::vector<MaskedTable *> input, output;
            std::vector<SharedSortHandle> handles;
            std::unique_ptr<MaskedTable> combined;
            std::unique_ptr<MaskedSortGroup<std::uint64_t>> sort;

            TimeGroup(std::size_t n) : input(n, nullptr), output(n, nullptr), handles(n) {
            }

            void prepare(MaterialStore &s, const MaterialPlan &p, std::size_t id) {
                std::vector<unsigned> widths{64};
                for (auto *table : input) {
                    if (!table) {
                        throw std::logic_error("missing shared time input");
                    }
                    for (std::size_t j = 1; j < table->columns.size(); ++j) {
                        widths.push_back(table->columns[j].width);
                    }
                }
                combined = std::make_unique<MaskedTable>(input[0]->columns[0].count, widths);
                std::array<MaskedTable *, 1> tables{combined.get()};
                std::array<std::string, 1> names{"multi.time." + std::to_string(id)};
                sort = std::make_unique<MaskedSortGroup<std::uint64_t>>(tables, s, p, names, 64);
            }

            void run(Net &net) {
                Timer timer;
                copy_column(input[0]->columns[0], combined->columns[0]);
                std::size_t col = 1;
                for (auto *table : input) {
                    for (std::size_t j = 1; j < table->columns.size(); ++j) {
                        copy_column(table->columns[j], combined->columns[col++]);
                    }
                }
                combined->dealer_knows_positions = false;
                sort->run(net);
                auto &sorted = sort->output();
                col = 1;
                for (std::size_t i = 0; i < output.size(); ++i) {
                    copy_column(sorted.columns[0], output[i]->columns[0]);
                    for (std::size_t j = 1; j < output[i]->columns.size(); ++j) {
                        copy_column(sorted.columns[col++], output[i]->columns[j]);
                    }
                    output[i]->mark_public_reordered();
                    handles[i].stats = sort->sort_stats[0];
                    handles[i].seconds = timer.seconds();
                }
            }
        };
    }

    struct MultiQuery::Impl : OpenAugment {
        Plan cfg;
        MaterialStore &store;
        const MaterialPlan &plan;
        const InputManager &input;
        MultiCompilation compilation;
        RowUnion row_union;
        std::unique_ptr<Compact> compact;
        std::vector<std::unique_ptr<TemplateRun>> runs;
        std::vector<std::unique_ptr<TimeGroup>> groups;
        std::unique_ptr<MultiRound> scheduler;
        std::vector<std::unique_ptr<ThresholdGroup>> thresholds;
        std::vector<OpenBuf *> extra_opens;
        std::vector<ThresholdGroup *> pending;
        StageStats union_stats, compact_stats, operator_stats;
        bool used = false;

        Impl(const Plan &c, MaterialStore &s, const MaterialPlan &p, const InputManager &in)
            : cfg(c), store(s), plan(p), input(in), compilation(compile_multi(c)),
              row_union(cfg, s, p, in), union_stats(cfg.common.n), compact_stats(cfg.common.n),
              operator_stats(cfg.common.n) {
            if (compilation.material.fingerprint() != p.fingerprint()) {
                throw std::invalid_argument("multi material/plan mismatch");
            }
            compact = std::make_unique<Compact>(cfg.common, s, p, in, row_union.result->words());
            for (const auto &members : compilation.group_members) {
                groups.push_back(std::make_unique<TimeGroup>(members.size()));
            }
            for (const auto &members : compilation.interval_members) {
                thresholds.push_back(std::make_unique<ThresholdGroup>(members.size()));
            }
            extra_opens.reserve(thresholds.size());
            pending.reserve(thresholds.size());
            const bool align_nl =
                std::any_of(cfg.templates.begin(), cfg.templates.end(), [](auto &q) {
                    return q.config.algorithm == "nl" && !q.filter.empty();
                });
            for (std::size_t i = 0; i < cfg.templates.size(); ++i) {
                SortBinder binder = [&, i](const std::string &name,
                                           MaskedTable &table,
                                           MaskedTable &output) -> SharedSortHandle * {
                    const auto g = compilation.time_groups[i];
                    if (name != "pdc.time" || g < 0) {
                        return nullptr;
                    }
                    const auto &members = compilation.group_members[static_cast<std::size_t>(g)];
                    const auto j = static_cast<std::size_t>(
                        std::find(members.begin(), members.end(), i) - members.begin());
                    auto &group = *groups[static_cast<std::size_t>(g)];
                    group.input[j] = &table;
                    group.output[j] = &output;
                    group.handles[j].job = cfg.templates.size() + static_cast<std::size_t>(g);
                    return &group.handles[j];
                };
                IntervalBinder interval_binding =
                    [&](std::size_t cmp,
                        unsigned bytes,
                        const void *input,
                        std::span<const BandWindow> windows,
                        std::span<const std::span<std::uint64_t>> outputs)
                    -> SharedIntervalHandle * {
                    for (std::size_t g = 0; g < thresholds.size(); ++g) {
                        const auto &members = compilation.interval_members[g];
                        const auto it = std::find(members.begin(), members.end(), cmp);
                        if (it == members.end()) {
                            continue;
                        }
                        const auto j = static_cast<std::size_t>(it - members.begin());
                        if (windows.size() != 1 || outputs.size() != 1 ||
                            thresholds[g]->registered[j]) {
                            throw std::invalid_argument(
                                "duplicate/invalid shared endpoint binding");
                        }
                        thresholds[g]->inputs[j] = input;
                        thresholds[g]->native_bytes[j] = bytes;
                        thresholds[g]->outputs[j] = outputs[0];
                        thresholds[g]->registered[j] = true;
                        return &thresholds[g]->handles[j];
                    }
                    return nullptr;
                };
                interval_binder(&interval_binding);
                sort_binder(&binder);
                try {
                    runs.push_back(std::make_unique<TemplateRun>(
                        cfg.templates[i], cfg.common, s, p, *compact, i, align_nl));
                } catch (...) {
                    sort_binder(nullptr);
                    interval_binder(nullptr);
                    throw;
                }
                sort_binder(nullptr);
                interval_binder(nullptr);
            }
            for (std::size_t g = 0; g < groups.size(); ++g) {
                groups[g]->prepare(s, p, g);
            }
            for (std::size_t g = 0; g < thresholds.size(); ++g) {
                const auto &members = compilation.interval_members[g];
                const auto root = p.comparisons[members[0]].table_source.empty()
                                      ? members[0]
                                      : named_comparison(p, p.comparisons[members[0]].table_source);
                const auto position = static_cast<std::size_t>(
                    std::find(members.begin(), members.end(), root) - members.begin());
                thresholds[g]->prepare(s, p, root, compilation.interval_windows[g], position);
            }
        }

        void prepare(Net &net, unsigned threads) {
            if (scheduler || used) {
                throw std::logic_error("multi schedule prepared twice");
            }
            std::vector<ScheduledJob> jobs;
            std::size_t bytes = 1024;
            for (auto &run : runs) {
                auto *ptr = run.get();
                jobs.push_back({[ptr, &net] {
                                    ptr->run(net);
                                },
                                {}});
                const auto &c = ptr->q.config;
                const auto bits =
                    c.algorithm == "nl" ? 512 * c.pairs() : 4096 * (c.x_size() + 2 * c.y_size());
                if (bits_bytes(bits) > SIZE_MAX - bytes) {
                    throw std::length_error("multi workspace overflow");
                }
                bytes += bits_bytes(bits);
            }
            for (std::size_t g = 0; g < groups.size(); ++g) {
                auto *ptr = groups[g].get();
                jobs.push_back({[ptr, &net] {
                                    ptr->run(net);
                                },
                                compilation.group_members[g]});
            }
            scheduler = std::make_unique<MultiRound>(
                net, std::move(jobs), bytes, 512 * runs.size(), threads, cfg.merge_rounds, this);
        }

        std::span<OpenBuf *const> prepare() override {
            extra_opens.clear();
            pending.clear();
            for (auto &group : thresholds) {
                if (auto *buf = group->request()) {
                    extra_opens.push_back(buf);
                    pending.push_back(group.get());
                }
            }
            return extra_opens;
        }

        void consume() override {
            for (auto *group : pending) {
                group->consume();
            }
            pending.clear();
        }

        void run(Net &net) {
            if (used || !scheduler) {
                throw std::logic_error("multi execution not prepared or reused");
            }
            used = true;
            Timer timer;
            row_union.run(net);
            union_stats.capture(net, timer.seconds());
            timer.reset();
            compact->run(net);
            compact_stats.capture(net, timer.seconds(), &union_stats);
            timer.reset();
            scheduler->run();
            operator_stats.capture(net, timer.seconds());
            operator_stats.open.rounds -= union_stats.open.rounds + compact_stats.open.rounds;
            operator_stats.open.hops -= union_stats.open.hops + compact_stats.open.hops;
            operator_stats.open.contribution_bits -=
                union_stats.open.contribution_bits + compact_stats.open.contribution_bits;
            for (std::size_t i = 0; i < operator_stats.peers.size(); ++i) {
                auto &p = operator_stats.peers[i];
                const auto &u = union_stats.peers[i];
                const auto &c = compact_stats.peers[i];
                p.sent_payload -= u.sent_payload + c.sent_payload;
                p.received_payload -= u.received_payload + c.received_payload;
                p.sent_wire -= u.sent_wire + c.sent_wire;
                p.received_wire -= u.received_wire + c.received_wire;
                p.sent_messages -= u.sent_messages + c.sent_messages;
                p.received_messages -= u.received_messages + c.received_messages;
            }
        }
    };

    MultiQuery::MultiQuery(const Plan &p,
                           MaterialStore &s,
                           const MaterialPlan &m,
                           const InputManager &i)
        : impl_(std::make_unique<Impl>(p, s, m, i)) {
    }

    MultiQuery::~MultiQuery() = default;

    void MultiQuery::prepare(Net &n, unsigned threads) {
        impl_->prepare(n, threads);
    }

    void MultiQuery::run(Net &n) {
        impl_->run(n);
    }

    const MergeStats &MultiQuery::stats() const {
        return impl_->scheduler->stats();
    }

    std::span<const std::uint64_t> MultiQuery::counts(std::size_t i) const {
        return impl_->runs.at(i)->counts();
    }

    std::span<const std::uint64_t> MultiQuery::alarms(std::size_t i) const {
        return impl_->runs.at(i)->alarms();
    }

    std::uint64_t MultiQuery::overflow() const {
        for (std::size_t i = 0; i < impl_->cfg.templates.size(); ++i) {
            if (impl_->cfg.templates[i].config.pattern != "P3") {
                return impl_->runs[i]->overflow();
            }
        }
        return impl_->runs[0]->overflow();
    }

    const CompactStats &MultiQuery::compact_stats() const {
        return impl_->compact->stats();
    }
}

namespace bj {
    const StageStats &MultiQuery::union_stage() const {
        return impl_->union_stats;
    }

    const StageStats &MultiQuery::compact_stage() const {
        return impl_->compact_stats;
    }

    const StageStats &MultiQuery::operator_stage() const {
        return impl_->operator_stats;
    }

    double MultiQuery::template_seconds(std::size_t i) const {
        return impl_->runs.at(i)->elapsed;
    }
}
