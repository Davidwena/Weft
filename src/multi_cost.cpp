#include "bj/plan/multi.hpp"
#include "bj/proto/compare_shape.hpp"
#include <cmath>
#include <nlohmann/json.hpp>
#include <numeric>

namespace bj {
    using Json = nlohmann::json;

    namespace {
        std::uint64_t provision(const MaterialPlan &p) {
            MaterialLayout l(p, 0, MaterialMode::Random);
            auto bits = l.logical_share_bits;
            for (const auto &s : l.segments) {
                if (s.role == SegmentRole::ShufflePrivate && s.storage == SegmentStorage::Dense) {
                    bits += s.bits;
                }
            }
            std::vector<std::uint64_t> owned(p.n, 0);
            for (const auto &in : p.inputs) {
                owned[in.owner] += checked_bits(in.count, in.width ? in.width : 1);
            }
            return bits + *std::max_element(owned.begin(), owned.end());
        }

        double average_comparisons(std::size_t N) {
            double H = 0;
            for (std::size_t i = 1; i <= N; ++i) {
                H += 1.0 / static_cast<double>(i);
            }
            return N < 2 ? 0 : 2 * (static_cast<double>(N) + 1) * H - 4 * static_cast<double>(N);
        }

        unsigned depth(std::size_t N) {
            return N < 2 ? 0
                         : static_cast<unsigned>(
                               std::min(static_cast<double>(N - 1),
                                        std::ceil(4.31107 * std::log(static_cast<double>(N)))));
        }

        struct Trace {
            std::vector<int> stages;
            std::size_t pos = 0;
            bool active = true;
        };

        void opens(Trace &t, unsigned n) {
            t.stages.insert(t.stages.end(), n, 0);
        }

        void sort(Trace &t, std::size_t N) {
            opens(t, 1);
            t.stages.push_back(1);
            opens(t, 3 * depth(N));
        }

        std::pair<std::uint64_t, std::uint64_t> graph(const Plan &cfg,
                                                      const MultiCompilation &compilation) {
            const auto M = cfg.templates.size();
            const bool nl_preparation =
                std::any_of(cfg.templates.begin(), cfg.templates.end(), [](auto &q) {
                    return q.config.algorithm == "nl" && !q.filter.empty();
                });
            std::vector<Trace> jobs(M + compilation.group_members.size());
            for (std::size_t i = 0; i < M; ++i) {
                const auto &q = cfg.templates[i];
                const auto &c = q.config;
                auto &t = jobs[i];
                const auto N = c.x_size() + 2 * c.y_size();
                const auto h = std::bit_width(N ? N - 1 : 0);
                if (c.algorithm == "nl" && nl_preparation) {
                    opens(t, 1);
                } else if (c.algorithm == "dc" && !q.filter.empty()) {
                    opens(t, 2);
                }
                if (c.algorithm == "nl") {
                    opens(t, 2 + 3 * (c.pattern == "P2" || c.pattern == "P3"));
                } else if (!N || (c.pattern != "P3" && (!c.x_size() || !c.y_size()))) {
                    opens(t, 1 + 3 * (c.pattern == "P2" || c.pattern == "P3"));
                } else {
                    if (c.pattern == "P3") {
                        opens(t, 2);
                    }
                    if (c.pattern == "P4") {
                        sort(t, N);
                        opens(t, 3);
                    }
                    if (compilation.time_groups[i] >= 0) {
                        t.stages.push_back(2 + compilation.time_groups[i]);
                    } else {
                        sort(t, N);
                    }
                    if (c.pattern == "P3") {
                        sort(t, N);
                        opens(t, 3);
                    } else {
                        sort(t, N);
                        for (unsigned l = 0; l < h; ++l) {
                            opens(t, 2);
                            if (l + 1 < h) {
                                opens(t, 1);
                                t.stages.push_back(1);
                                opens(t, 1);
                            }
                        }
                        if (c.pattern == "P2") {
                            sort(t, N);
                            opens(t, 3);
                        }
                    }
                }
                if (c.algorithm == "dc" && c.pattern == "P2" && c.kout && !q.filter.empty()) {
                    opens(t, 1);
                }
                if (c.output == "alarm") {
                    opens(t, 2);
                }
            }
            for (std::size_t g = 0; g < compilation.group_members.size(); ++g) {
                auto &t = jobs[M + g];
                t.active = false;
                const auto &c = cfg.templates[compilation.group_members[g][0]].config;
                sort(t, c.x_size() + 2 * c.y_size());
            }
            std::uint64_t rounds = 0, shuffles = 0;
            for (;;) {
                bool changed = false;
                for (std::size_t g = 0; g < compilation.group_members.size(); ++g) {
                    auto &shared = jobs[M + g];
                    if (!shared.active && std::all_of(compilation.group_members[g].begin(),
                                                      compilation.group_members[g].end(),
                                                      [&](auto i) {
                                                          return jobs[i].pos <
                                                                     jobs[i].stages.size() &&
                                                                 jobs[i].stages[jobs[i].pos] ==
                                                                     static_cast<int>(2 + g);
                                                      })) {
                        shared.active = true;
                        changed = true;
                    }
                    if (shared.active && shared.pos == shared.stages.size()) {
                        for (auto i : compilation.group_members[g]) {
                            if (jobs[i].pos < jobs[i].stages.size() &&
                                jobs[i].stages[jobs[i].pos] == static_cast<int>(2 + g)) {
                                ++jobs[i].pos;
                                changed = true;
                            }
                        }
                    }
                }
                if (changed) {
                    continue;
                }
                if (std::all_of(jobs.begin(), jobs.end(), [](auto &t) {
                        return t.pos == t.stages.size();
                    })) {
                    break;
                }
                std::size_t no = 0, ns = 0;
                for (auto &t : jobs) {
                    if (t.active && t.pos < t.stages.size() && t.stages[t.pos] < 2) {
                        if (t.stages[t.pos] == 0) {
                            ++no;
                        } else {
                            ++ns;
                        }
                        ++t.pos;
                    }
                }
                if (!no && !ns) {
                    throw std::logic_error("cost dependency cycle");
                }
                rounds += cfg.merge_rounds ? static_cast<std::uint64_t>(no > 0) : no;
                const auto batches = cfg.merge_rounds ? static_cast<std::uint64_t>(ns > 0) : ns;
                rounds += batches * cfg.common.n;
                shuffles += batches;
            }
            return {rounds, shuffles};
        }

        Json estimate(const Plan &cfg, const NetworkModel &net) {
            const auto compilation = compile_multi(cfg);
            const auto &p = compilation.material;
            const auto &c = cfg.common;
            double open_bits = 0, shuffle_bytes = 0;
            for (const auto &cmp : p.comparisons) {
                if (!cmp.evaluation_sources.empty()) {
                    continue;
                }
                const bool msb = cmp.kind == CompareKind::MsbKnown ||
                                 cmp.kind == CompareKind::MsbFresh ||
                                 cmp.kind == CompareKind::Interval;
                const auto ell = msb ? cmp.width - 1 : cmp.width;
                CompareShape shape(ell,
                                   cmp.block_width(),
                                   cmp.kind == CompareKind::Eq || cmp.kind == CompareKind::EqFresh);
                const bool pool = cmp.name.ends_with(".compare");
                double count = static_cast<double>(cmp.count);
                if (pool) {
                    std::size_t N = 0;
                    if (cmp.name.starts_with("multi.time.")) {
                        auto g = static_cast<std::size_t>(std::stoul(cmp.name.substr(11)));
                        const auto &cc = cfg.templates[compilation.group_members[g][0]].config;
                        N = cc.x_size() + 2 * cc.y_size();
                    } else {
                        const auto qi = static_cast<std::size_t>(std::stoul(cmp.name.substr(1)));
                        const auto &cc = cfg.templates[qi].config;
                        N = cc.x_size() + 2 * cc.y_size();
                    }
                    count = average_comparisons(N);
                }
                open_bits +=
                    count *
                    (shape.inputs * cmp.evaluations +
                     ((cmp.kind == CompareKind::MsbFresh || cmp.kind == CompareKind::EqFresh)
                          ? cmp.width
                          : 0) +
                     (pool ? 1 : 0));
            }
            for (const auto &gate : p.gates) {
                if (gate.name != "output.omega") {
                    open_bits += static_cast<double>(gate.m) * static_cast<double>(gate.count);
                }
            }
            for (const auto &m : p.multiplications) {
                open_bits += 2.0 * m.width * static_cast<double>(m.count);
                if (m.width == 32) {
                    open_bits += 32.0 * static_cast<double>(m.count);
                }
            }
            for (const auto &m : p.bitmuls) {
                open_bits += static_cast<double>(m.count);
            }
            const auto rows = c.size() + c.kin + c.kout;
            open_bits += 64.0 * static_cast<double>(rows) + 2.0 * static_cast<double>(rows) +
                         (161 + (c.carry_cp() ? 96 : 0) + c.filters.size()) *
                             static_cast<double>(c.kin + c.kout);
            for (const auto &m : p.masks) {
                if (m.fresh && (m.name.find(".mask.") != std::string::npos ||
                                (m.name.starts_with("q") && (m.name.ends_with("slot.vin") ||
                                                             m.name.ends_with("slot.vout"))))) {
                    open_bits +=
                        static_cast<double>(m.width ? m.width : 1) * static_cast<double>(m.count);
                }
            }
            for (const auto &s : p.shuffles) {
                for (auto id : s.inputs) {
                    const auto k = id < p.inputs.size() ? p.inputs[id].width
                                                        : p.masks[id - p.inputs.size()].width;
                    shuffle_bytes +=
                        static_cast<double>(bits_bytes(checked_bits(s.count, k ? k : 1)));
                }
            }
            auto [r, shuffle_batches] = graph(cfg, compilation);
            unsigned union_depth = 0;
            if (!c.filters.empty() &&
                std::none_of(cfg.templates.begin(), cfg.templates.end(), [](auto &q) {
                    return q.filter.empty();
                })) {
                union_depth = static_cast<unsigned>(std::bit_width(c.filters.size() - 1));
            }

            const auto shuffle_hops = (shuffle_batches + 1) * c.n;
            const auto open_rounds = r - shuffle_batches * c.n + 11 + union_depth;
            const bool king = c.n >= 3 && c.open_mode == "auto";
            const double open_factor = king ? 2.0 * (c.n - 1) : static_cast<double>(c.n - 1);
            const double open_hops = king ? 2.0 : 1.0;
            const double open_seconds =
                static_cast<double>(open_rounds) * open_hops * net.one_way_seconds +
                open_factor * open_bits / 8 / net.bytes_per_second;
            const double shuffle_seconds = static_cast<double>(shuffle_hops) * net.one_way_seconds +
                                           2.0 * (c.n - 1) * shuffle_bytes / net.bytes_per_second;
            std::vector<double> owner_bytes(c.n, 0);
            double input_total = 0;
            for (const auto &in : p.inputs) {
                const auto b = static_cast<double>(
                    bits_bytes(checked_bits(in.count, in.width ? in.width : 1)));
                owner_bytes[in.owner] += b;
                input_total += b;
            }
            double bottleneck = 0;
            for (auto b : owner_bytes) {
                bottleneck = std::max(bottleneck, std::max((c.n - 1) * b, input_total - b));
            }
            const auto input_seconds = net.one_way_seconds + bottleneck / net.bytes_per_second;
            double output_bytes = 8;
            for (const auto &q : cfg.templates) {
                output_bytes += q.config.output == "count" ? 8 : 1;
            }
            const auto output_seconds =
                net.one_way_seconds + (c.n - 1) * output_bytes / net.bytes_per_second;
            auto base = c;
            base.mode = "compact";
            auto baseplan = compile_investigation(base);
            if (!c.filters.empty() && union_depth) {
                for (std::size_t i = 0; i < c.filters.size() - 1; ++i) {
                    baseplan.gates.push_back(
                        {"model.union." + std::to_string(i), 2, 32, c.size(), {}});
                }
            }
            const auto compact = provision(baseplan), total = provision(p);
            return {{"estimated_seconds",
                     open_seconds + shuffle_seconds + input_seconds + output_seconds},
                    {"open_seconds", open_seconds},
                    {"shuffle_seconds", shuffle_seconds},
                    {"input_seconds", input_seconds},
                    {"private_output_seconds", output_seconds},
                    {"estimated_open_rounds", open_rounds},
                    {"estimated_shuffle_hops", shuffle_hops},
                    {"estimated_operator_graph_rounds", r},
                    {"estimated_open_contribution_bits", open_bits},
                    {"shuffle_table_bytes", shuffle_bytes},
                    {"preprocessing_bits", total},
                    {"compact_preprocessing_bits", compact},
                    {"operator_preprocessing_bits", total >= compact ? total - compact : 0},
                    {"shared_threshold_tables", compilation.shared_threshold_tables},
                    {"shared_time_groups", compilation.group_members.size()}};
        }
    }

    Json
    enumerate_operators(const Plan &original, const NetworkModel &network, std::uint64_t budget) {
        original.validate();
        if (!std::isfinite(network.one_way_seconds) || network.one_way_seconds < 0 ||
            !std::isfinite(network.bytes_per_second) || network.bytes_per_second <= 0) {
            throw std::invalid_argument("invalid network model");
        }
        const auto M = original.templates.size();
        Json candidates = Json::array(), best = nullptr;
        double fastest = INFINITY;
        const auto combinations = std::uint64_t{1} << M;
        for (std::uint64_t mask = 0; mask < combinations; ++mask) {
            auto cfg = original;
            Json choices = Json::array();
            for (std::size_t i = 0; i < M; ++i) {
                cfg.templates[i].config.algorithm = (mask >> i) & 1 ? "dc" : "nl";
                choices.push_back(cfg.templates[i].config.algorithm);
            }
            try {
                auto row = estimate(cfg, network);
                row["mask"] = mask;
                row["operators"] = choices;
                const bool feasible = row.at("preprocessing_bits").get<std::uint64_t>() <= budget;
                row["feasible"] = feasible;
                if (feasible && row["estimated_seconds"].get<double>() < fastest) {
                    fastest = row["estimated_seconds"];
                    best = mask;
                }
                candidates.push_back(row);
            } catch (const std::invalid_argument &e) {
                candidates.push_back({{"mask", mask},
                                      {"operators", choices},
                                      {"feasible", false},
                                      {"reason", e.what()}});
            }
        }
        return {{"selected_mask", best},
                {"budget_bits", budget},
                {"budget_scope",
                 "per-party logical random material, including Compact/input shares and "
                 "shuffle private corrections/permutations and maximum owned full input "
                 "masks"},
                {"one_way_seconds", network.one_way_seconds},
                {"bytes_per_second", network.bytes_per_second},
                {"sort_estimate",
                 "expected random-quicksort comparisons; bounded "
                 "ceil(4.31107 ln N) depth, not a measured guarantee"},
                {"candidates", candidates}};
    }
}
