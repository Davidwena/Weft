#include "bj/core/material.hpp"
#include <nlohmann/json.hpp>
#include <cstdio>

int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("plan_info plan.json");
        }
        const auto p = bj::MaterialPlan::load(argv[1]);
        nlohmann::json result;
        result["plan_fingerprint"] = p.fingerprint();
        result["parties"] = nlohmann::json::array();
        for (auto mode : {bj::MaterialMode::Random, bj::MaterialMode::Prg}) {
            for (unsigned id = 0; id < p.n; ++id) {
                bj::MaterialLayout l(p, id, mode);
                nlohmann::json by_role = nlohmann::json::object();
                for (const auto &s : l.segments) {
                    auto key = std::to_string(static_cast<unsigned>(s.role));
                    if (!by_role.contains(key)) {
                        by_role[key] = 0;
                    }
                    by_role[key] = by_role[key].get<std::uint64_t>() + s.bits;
                }
                result["parties"].push_back({{"party", id},
                                             {"mode", bj::material_mode_name(mode)},
                                             {"file_bytes", l.file_bytes},
                                             {"logical_share_bits", l.logical_share_bits},
                                             {"gate_bits", l.gate_bits},
                                             {"dense_share_bytes", l.dense_share_bytes},
                                             {"owner_bytes", l.owner_bytes},
                                             {"seed_bytes", l.seed_bytes},
                                             {"role_bits", by_role}});
            }
        }
        result["sort_pools"] = nlohmann::json::array();
        for (const auto &c : p.comparisons) {
            if (c.name.starts_with("dc.sort.") || c.name.ends_with(".compare")) {
                result["sort_pools"].push_back(
                    {{"name", c.name}, {"width", c.width}, {"count", c.count}});
            }
        }
        std::puts(result.dump().c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "plan_info: %s\n", e.what());
        return 2;
    }
}
