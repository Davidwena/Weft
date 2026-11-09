#include "bj/ops/compact.hpp"
#include "bj/ops/quicksort.hpp"
#include "bj/plan/multi.hpp"
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>

namespace bj {
    using Json = nlohmann::json;

    namespace {
        bool identifier(const std::string &s) {
            return !s.empty() && s.size() <= 32 &&
                   std::all_of(s.begin(), s.end(), [](unsigned char ch) {
                       return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                              (ch >= '0' && ch <= '9') || ch == '_';
                   });
        }

        bool core(const std::string &s) {
            return s.starts_with("pair.") || s.starts_with("band.") || s.starts_with("pattern.") ||
                   s.starts_with("pdc.") || s == "output.omega";
        }

        std::string prefix(std::size_t i) {
            return "q" + std::to_string(i) + ".";
        }

        std::string derived_key(const MaskSpec &m) {
            Json terms = Json::array();
            for (const auto &t : m.terms) {
                terms.push_back({t.source,
                                 t.source_offset,
                                 t.destination_offset,
                                 t.length,
                                 t.divisor,
                                 t.modulus,
                                 t.coefficient});
            }
            return Json{m.width, m.count, terms}.dump();
        }
    }

    Plan Plan::parse(const Json &document) {
        const auto &j = document.at("multi");
        Plan p;
        auto common = j.at("common");
        common["mode"] = "pipeline";
        common["pattern"] = "P1";
        common["operator"] = "nl";
        common.erase("carried_filters");
        p.common = InvestigationConfig::parse({{"investigation", common}});
        p.merge_rounds = j.value("merge_rounds", true);
        p.share_thresholds = j.value("share_thresholds", true);
        p.share_time_sorts = j.value("share_time_sorts", true);
        for (const auto &t : j.at("templates")) {
            auto c = common;
            for (auto it = t.begin(); it != t.end(); ++it) {
                if (it.key() != "name" && it.key() != "filter") {
                    static const std::set<std::string> allowed{"pattern",
                                                               "window",
                                                               "windows",
                                                               "p",
                                                               "q",
                                                               "a_lo",
                                                               "a_hi",
                                                               "m",
                                                               "operator",
                                                               "output",
                                                               "threshold",
                                                               "dc_pool_factor",
                                                               "dc_pool_safe"};
                    if (!allowed.contains(it.key())) {
                        throw std::invalid_argument("unsupported template field: " + it.key());
                    }
                    if (it.key() == "window") {
                        c["windows"] = Json::array({it.value()});
                    } else {
                        c[it.key()] = it.value();
                    }
                }
            }
            QueryTemplate q;
            q.name = t.at("name");
            q.filter = t.value("filter", std::string{});
            q.config = InvestigationConfig::parse({{"investigation", c}});
            p.templates.push_back(std::move(q));
        }
        std::set<std::string> filters;
        for (const auto &t : p.templates) {
            if (!t.filter.empty()) {
                filters.insert(t.filter);
            }
        }
        p.common.filters.assign(filters.begin(), filters.end());
        if (std::any_of(p.templates.begin(), p.templates.end(), [](auto &t) {
                return t.config.carry_cp();
            })) {
            p.common.pattern = "P4";
        }
        p.validate();
        return p;
    }

    Plan Plan::load(const std::filesystem::path &path) {
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("cannot read multi plan");
        }
        auto j = Json::parse(f);
        if (j.contains("context")) {
            j = Json::parse(j.at("context").get<std::string>());
        }
        return parse(j);
    }

    void Plan::validate() const {
        common.validate();
        std::set<std::string> needed_filters;
        bool need_cp = false;
        for (const auto &q : templates) {
            if (!q.filter.empty()) {
                needed_filters.insert(q.filter);
            }
            need_cp = need_cp || q.config.carry_cp();
        }
        if (std::vector<std::string>(needed_filters.begin(), needed_filters.end()) !=
                common.filters ||
            common.carry_cp() != need_cp) {
            throw std::invalid_argument("shared filter/counterparty schema mismatch");
        }
        if (common.mode != "pipeline" || templates.empty() || templates.size() > 16) {
            throw std::invalid_argument("multi requires 1..16 pipeline templates");
        }
        std::set<std::string> names;
        for (const auto &q : templates) {
            q.config.validate();
            if (!identifier(q.name) || !names.insert(q.name).second ||
                (!q.filter.empty() && !identifier(q.filter)) || q.config.windows.size() != 1 ||
                q.config.variant != "C1+C2" || q.config.n != common.n ||
                q.config.kin != common.kin || q.config.kout != common.kout ||
                q.config.rows != common.rows || q.config.initiator != common.initiator ||
                q.config.king != common.king || q.config.open_mode != common.open_mode ||
                q.config.bt != common.bt || q.config.ba != common.ba ||
                q.config.mode != "pipeline") {
                throw std::invalid_argument("template does not match shared pipeline");
            }
        }
    }

    Json Plan::json() const {
        Json ts = Json::array();
        for (const auto &t : templates) {
            auto j = t.config.json()["investigation"];
            Json q = {{"name", t.name},
                      {"pattern", t.config.pattern},
                      {"operator", t.config.algorithm},
                      {"window", {t.config.windows[0].lower, t.config.windows[0].upper}},
                      {"p", t.config.p},
                      {"q", t.config.q},
                      {"a_lo", t.config.amount_lower},
                      {"a_hi", t.config.amount_upper},
                      {"m", t.config.min_cluster},
                      {"output", t.config.output},
                      {"threshold", t.config.threshold},
                      {"dc_pool_factor", t.config.dc_pool_factor},
                      {"dc_pool_safe", t.config.dc_pool_safe}};
            if (!t.filter.empty()) {
                q["filter"] = t.filter;
            }
            ts.push_back(q);
        }
        return {{"multi",
                 {{"common", common.json()["investigation"]},
                  {"templates", ts},
                  {"merge_rounds", merge_rounds},
                  {"share_thresholds", share_thresholds},
                  {"share_time_sorts", share_time_sorts}}}};
    }

    MultiCompilation compile_multi(const Plan &cfg) {
        cfg.validate();
        MultiCompilation result;
        auto base = cfg.common;
        base.mode = "compact";
        base.algorithm = "nl";
        result.material = compile_investigation(base);
        auto &p = result.material;
        auto context = cfg.json();
        context["investigation"] = cfg.common.json()["investigation"];
        p.context = context.dump();
        const auto M = cfg.templates.size();
        result.time_groups.assign(M, -1);
        if (cfg.share_time_sorts) {
            std::map<std::string, std::vector<std::size_t>> groups;
            for (std::size_t i = 0; i < M; ++i) {
                const auto &c = cfg.templates[i].config;
                const auto N = c.x_size() + 2 * c.y_size();
                if (c.algorithm != "dc" || !N ||
                    (c.pattern != "P3" && (!c.x_size() || !c.y_size()))) {
                    continue;
                }
                auto family = c.pattern == "P1" || c.pattern == "P2" ? "P12" : c.pattern;
                groups[Json{family, c.kin, c.kout, c.bt, c.windows[0].lower, c.windows[0].upper}
                           .dump()]
                    .push_back(i);
            }
            for (const auto &[key, members] : groups) {
                (void)key;
                if (members.size() < 2) {
                    continue;
                }
                const auto g = result.group_members.size();
                result.group_members.push_back(members);
                for (auto i : members) {
                    result.time_groups[i] = static_cast<int>(g);
                }
            }
        }
        if (!cfg.common.filters.empty() &&
            std::none_of(cfg.templates.begin(), cfg.templates.end(), [](auto &q) {
                return q.filter.empty();
            })) {
            for (std::size_t level = 0, count = cfg.common.filters.size(); count > 1; ++level) {
                for (std::size_t j = 0; j < count / 2; ++j) {
                    p.gates.push_back(
                        {"multi.union." + std::to_string(level) + "." + std::to_string(j),
                         2,
                         32,
                         cfg.common.size(),
                         {}});
                }
                count = (count + 1) / 2;
            }
        }
        std::map<std::string, std::size_t> derived;
        for (std::size_t i = 0; i < p.masks.size(); ++i) {
            if (!p.masks[i].fresh) {
                derived[derived_key(p.masks[i])] = p.inputs.size() + i;
            }
        }
        for (std::size_t qi = 0; qi < M; ++qi) {
            const auto &query = cfg.templates[qi];
            const auto &c = query.config;
            const auto pre = prefix(qi);
            if (!query.filter.empty()) {
                if (c.algorithm == "nl") {
                    p.gates.push_back({pre + "extra.pair", 2, 32, c.pairs(), {}});
                } else {
                    for (auto side : {"in", "out"}) {
                        const auto n = std::string(side) == "in" ? c.kin : c.kout;
                        p.gates.push_back({pre + "extra." + side, 2, 32, n, {}});
                        p.masks.push_back({pre + "slot.v" + side, 0, n, true, true, {}});
                    }
                }
            }
            if (!query.filter.empty() && c.algorithm == "dc" && c.pattern == "P2") {
                p.gates.push_back({pre + "pattern.query_filter", 2, 32, c.kout, {}});
            }
            auto cc = c;
            cc.filters.clear();
            cc.mode = "compact";
            auto local = compile_investigation(cc);
            compile_pattern_material(c, local);
            if (c.pattern == "P1" && c.algorithm == "nl") {
                for (auto &cmp : local.comparisons) {
                    if (cmp.name == "pattern.alarm") {
                        cmp.name = "band.alarm";
                    }
                }
            }
            std::vector<std::size_t> refs(local.inputs.size() + local.masks.size(), SIZE_MAX);
            for (std::size_t i = 0; i < local.inputs.size(); ++i) {
                refs[i] = named_mask(p, local.inputs[i].name);
            }
            for (std::size_t i = 0; i < local.masks.size(); ++i) {
                auto m = local.masks[i];
                const auto old = local.inputs.size() + i;
                if (!core(m.name)) {
                    auto name = m.name;
                    if (!query.filter.empty() && c.algorithm == "dc" &&
                        (name == "slot.vin" || name == "slot.vout")) {
                        name = pre + name;
                    }
                    refs[old] = named_mask(p, name);
                    continue;
                }
                if (result.time_groups[qi] >= 0 && m.name.starts_with("pdc.time.")) {
                    continue;
                }
                for (auto &t : m.terms) {
                    t.source = refs.at(t.source);
                    if (t.source == SIZE_MAX) {
                        throw std::logic_error("unmapped shared sort mask dependency");
                    }
                }
                const auto key = derived_key(m);
                if (!m.fresh && derived.contains(key)) {
                    refs[old] = derived.at(key);
                    continue;
                }
                m.name = pre + m.name;
                refs[old] = p.inputs.size() + p.masks.size();
                if (!m.fresh) {
                    derived[key] = refs[old];
                }
                p.masks.push_back(std::move(m));
            }
            for (auto cmp : local.comparisons) {
                if (core(cmp.name)) {
                    if (result.time_groups[qi] >= 0 && cmp.name == "pdc.time.compare") {
                        continue;
                    }
                    for (auto &id : cmp.inputs) {
                        id = refs.at(id);
                    }
                    cmp.name = pre + cmp.name;
                    if (cfg.share_thresholds && cmp.kind == CompareKind::Interval) {
                        for (const auto &prev : p.comparisons) {
                            if (prev.kind == cmp.kind && prev.table_source.empty() &&
                                prev.width == cmp.width && prev.count == cmp.count &&
                                prev.block_width() == cmp.block_width() &&
                                prev.inputs == cmp.inputs) {
                                cmp.table_source = prev.name;
                                ++result.shared_threshold_tables;
                                break;
                            }
                        }
                    }
                    p.comparisons.push_back(std::move(cmp));
                }
            }
            for (auto gate : local.gates) {
                if (core(gate.name)) {
                    if (!query.filter.empty() && c.algorithm == "nl" &&
                        (gate.name.starts_with("band.product") || gate.name == "pattern.pair" ||
                         (c.pattern == "P3" && gate.name == "pattern.filter"))) {
                        ++gate.m;
                    }
                    gate.name = pre + gate.name;
                    p.gates.push_back(std::move(gate));
                }
            }
            for (auto v : local.bitmuls) {
                v.name = pre + v.name;
                v.mask = refs.at(v.mask);
                p.bitmuls.push_back(std::move(v));
            }
            for (auto v : local.multiplications) {
                v.name = pre + v.name;
                p.multiplications.push_back(std::move(v));
            }
            for (auto v : local.shuffles) {
                if (core(v.name)) {
                    if (result.time_groups[qi] >= 0 && v.name == "pdc.time") {
                        continue;
                    }
                    v.name = pre + v.name;
                    for (auto &id : v.inputs) {
                        id = refs.at(id);
                    }
                    for (auto &id : v.outputs) {
                        id = refs.at(id);
                    }
                    p.shuffles.push_back(std::move(v));
                }
            }
        }
        for (std::size_t g = 0; g < result.group_members.size(); ++g) {
            const auto &members = result.group_members[g];
            const auto &c = cfg.templates[members[0]].config;
            const auto N = c.x_size() + 2 * c.y_size();
            std::vector<unsigned> widths{64};
            for (auto i : members) {
                const auto &pattern = cfg.templates[i].config.pattern;
                const auto tail = pattern == "P3"   ? std::vector<unsigned>{96, 64}
                                  : pattern == "P4" ? std::vector<unsigned>{96, 96, 96, 0, 0, 0, 96}
                                                    : std::vector<unsigned>{96, 96, 96, 0, 0, 0};
                widths.insert(widths.end(), tail.begin(), tail.end());
            }
            const auto name = "multi.time." + std::to_string(g);
            std::vector<std::size_t> input, output;
            for (std::size_t j = 0; j < widths.size(); ++j) {
                input.push_back(p.inputs.size() + p.masks.size());
                p.masks.push_back(
                    {name + ".mask." + std::to_string(j), widths[j], N, true, true, {}});
                output.push_back(p.inputs.size() + p.masks.size());
                p.masks.push_back(
                    {name + ".out." + std::to_string(j), widths[j], N, true, true, {}});
            }
            p.shuffles.push_back({name, N, input, output});
            CompareSpec cmp;
            cmp.name = name + ".compare";
            cmp.kind = CompareKind::MsbFresh;
            cmp.width = 64;
            double pool_factor = 0;
            bool safe_pool = false;
            for (auto member : members) {
                const auto &request = cfg.templates[member].config;
                pool_factor = std::max(pool_factor, request.dc_pool_factor);
                safe_pool = safe_pool || request.dc_pool_safe;
            }
            cmp.count = dc_pool_capacity(N, pool_factor, safe_pool);
            p.comparisons.push_back(cmp);
        }
        if (cfg.share_thresholds) {
            std::map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>> candidates;
            for (std::size_t qi = 0; qi < M; ++qi) {
                const auto &c = cfg.templates[qi].config;
                if (c.algorithm != "nl") {
                    continue;
                }
                const auto id = named_comparison(
                    p,
                    prefix(qi) + (c.pattern == "P3" ? "pattern.time.interval" : "band.interval"));
                const auto root = p.comparisons[id].table_source.empty()
                                      ? id
                                      : named_comparison(p, p.comparisons[id].table_source);
                candidates[root].push_back({id, qi});
            }
            for (const auto &[root, entries] : candidates) {
                if (entries.size() < 2 || std::none_of(entries.begin(), entries.end(), [&](auto e) {
                        return e.first == root;
                    })) {
                    continue;
                }
                std::vector<std::size_t> members;
                std::vector<BandWindow> windows;
                std::vector<std::uint64_t> offsets;
                for (auto [id, qi] : entries) {
                    members.push_back(id);
                    const auto w = cfg.templates[qi].config.windows[0];
                    windows.push_back(w);
                    offsets.push_back(w.lower);
                    offsets.push_back(w.upper + 1);
                }
                std::sort(offsets.begin(), offsets.end());
                offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
                p.comparisons[root].evaluations = static_cast<unsigned>(offsets.size());
                for (std::size_t j = 0; j < members.size(); ++j) {
                    if (members[j] != root) {
                        auto &alias = p.comparisons[members[j]];
                        alias.evaluation_sources = {
                            static_cast<unsigned>(
                                std::lower_bound(offsets.begin(), offsets.end(), windows[j].lower) -
                                offsets.begin()),
                            static_cast<unsigned>(std::lower_bound(offsets.begin(),
                                                                   offsets.end(),
                                                                   windows[j].upper + 1) -
                                                  offsets.begin())};
                    }
                }
                result.interval_members.push_back(std::move(members));
                result.interval_windows.push_back(std::move(windows));
            }
        }
        p.validate();
        return result;
    }

    MaterialPlan template_material_view(const MaterialPlan &p, std::size_t index) {
        auto result = p;
        const auto pre = prefix(index);
        auto rename = [&](auto &entries) {
            for (auto &x : entries) {
                if (x.name.starts_with(pre)) {
                    x.name = x.name.substr(pre.size());
                } else {
                    x.name = "unused." + x.name;
                }
            }
        };
        rename(result.gates);
        rename(result.comparisons);
        rename(result.multiplications);
        rename(result.bitmuls);
        rename(result.shuffles);
        for (auto &m : result.masks) {
            if (m.name.starts_with(pre)) {
                m.name = m.name.substr(pre.size());
            }
        }
        return result;
    }
}
