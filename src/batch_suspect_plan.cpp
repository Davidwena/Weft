#include "bj/ops/batch_suspects.hpp"
#include <nlohmann/json.hpp>
#include <map>

namespace bj {
    namespace {
        std::string prefix(std::size_t j) {
            return "suspect" + std::to_string(j) + ".";
        }
    }

    MaterialPlan compile_batch_suspects(const InvestigationConfig &c, std::size_t S) {
        c.validate();
        if (!S || S > 64 || c.mode != "pipeline" || !c.filters.empty() || c.windows.size() != 1 ||
            c.output != "count") {
            throw std::invalid_argument(
                "batch requires 1..64 pipeline suspects, one window, count output, no filters");
        }
        const auto single = compile_investigation(c);
        MaterialPlan p;
        p.n = c.n;
        p.king = c.king;
        p.output_party = c.initiator;
        p.open_mode = c.open_mode;
        auto context = c.json();
        context["num_suspects"] = S;
        context["batch_path"] = true;
        p.context = context.dump();
        for (auto v : single.inputs) {
            if (v.name != "subject") {
                p.inputs.push_back(v);
            }
        }
        for (std::size_t j = 0; j < S; ++j) {
            p.inputs.push_back({prefix(j) + "subject", c.initiator, 64, 1});
        }
        std::map<std::string, std::size_t> shared;
        const auto M = c.size() + S * (c.kin + c.kout);
        for (std::size_t j = 0; j < S; ++j) {
            auto pre = prefix(j);
            std::vector<std::size_t> refs(single.inputs.size() + single.masks.size(), SIZE_MAX);
            for (std::size_t i = 0; i < single.inputs.size(); ++i) {
                refs[i] = named_mask(p,
                                     single.inputs[i].name == "subject" ? pre + "subject"
                                                                        : single.inputs[i].name);
            }
            for (std::size_t i = 0; i < single.masks.size(); ++i) {
                auto v = single.masks[i];
                const auto old = single.inputs.size() + i;
                const bool common = v.name.starts_with("table.") || v.name.starts_with("shuffle.");
                if (common && shared.contains(v.name)) {
                    refs[old] = shared.at(v.name);
                    continue;
                }
                for (auto &t : v.terms) {
                    t.source = refs.at(t.source);
                    if (t.source == SIZE_MAX) {
                        throw std::logic_error("batch mask topological ordering");
                    }
                }
                const auto name = v.name;
                if (common) {
                    v.count = M;
                } else {
                    v.name = pre + v.name;
                }
                refs[old] = p.inputs.size() + p.masks.size();
                if (common) {
                    shared[name] = refs[old];
                }
                p.masks.push_back(std::move(v));
            }
            for (auto v : single.comparisons) {
                if (!v.table_source.empty()) {
                    v.table_source = pre + v.table_source;
                }
                v.name = pre + v.name;
                for (auto &id : v.inputs) {
                    id = refs.at(id);
                }
                p.comparisons.push_back(std::move(v));
            }
            for (auto v : single.gates) {
                v.name = pre + v.name;
                for (auto &id : v.inputs) {
                    id = refs.at(id);
                }
                p.gates.push_back(std::move(v));
            }
            for (auto v : single.multiplications) {
                v.name = pre + v.name;
                p.multiplications.push_back(std::move(v));
            }
            for (auto v : single.bitmuls) {
                v.name = pre + v.name;
                v.mask = refs.at(v.mask);
                p.bitmuls.push_back(std::move(v));
            }
            for (auto v : single.shuffles) {
                if (v.name == "compact") {
                    if (j) {
                        continue;
                    }
                    v.count = M;
                } else {
                    v.name = pre + v.name;
                }
                for (auto &id : v.inputs) {
                    id = refs.at(id);
                }
                for (auto &id : v.outputs) {
                    id = refs.at(id);
                }
                p.shuffles.push_back(std::move(v));
            }
        }
        p.validate();
        return p;
    }

    MaterialPlan batch_suspect_view(const MaterialPlan &p, std::size_t j) {
        auto v = p;
        auto pre = prefix(j);
        auto rename = [&](auto &entries, bool common) {
            for (auto &e : entries) {
                if (e.name.starts_with(pre)) {
                    e.name = e.name.substr(pre.size());
                } else if (!common || e.name.starts_with("suspect")) {
                    e.name = "unused." + e.name;
                }
            }
        };
        rename(v.inputs, true);
        rename(v.masks, true);
        rename(v.comparisons, false);
        rename(v.gates, false);
        rename(v.bitmuls, false);
        rename(v.multiplications, false);
        rename(v.shuffles, true);
        return v;
    }
}
