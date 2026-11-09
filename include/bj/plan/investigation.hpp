#pragma once
#include "bj/core/material.hpp"
#include <nlohmann/json_fwd.hpp>

namespace bj {
    struct BandWindow {
        std::uint64_t lower = 0, upper = 1800;
    };

    struct InvestigationConfig {
        unsigned n = 3, king = 0, initiator = 0, bt = 29, ba = 40, bq = 7, kt = 32, ka = 49,
                 kp = 22;
        std::vector<std::size_t> rows;
        std::size_t kin = 512, kout = 512;

        std::size_t p3_query_begin = 0, p3_query_count = SIZE_MAX;
        std::uint64_t p = 5, q = 100, threshold = 1, amount_lower = 1, amount_upper = (1ull << 40),
                      min_cluster = 3;
        std::string pattern = "P1";
        std::vector<std::string> filters;

        bool carry_cp() const {
            return pattern == "P4";
        }

        std::size_t x_size() const {
            return pattern == "P4" ? kout : kin;
        }

        std::size_t y_size() const {
            return pattern == "P3"   ? (p3_query_count == SIZE_MAX ? kin : p3_query_count)
                   : pattern == "P4" ? kin
                                     : kout;
        }

        std::string algorithm = "nl";
        double dc_pool_factor = 1.5;
        bool dc_pool_safe = false;
        std::string mode = "pipeline", variant = "C1+C2", open_mode = "auto", output = "count";
        std::vector<BandWindow> windows{{0, 1800}};
        static InvestigationConfig parse(const nlohmann::json &);
        static InvestigationConfig load(const std::filesystem::path &);
        nlohmann::json json() const;
        void validate() const;
        std::size_t size() const;
        std::size_t pairs() const;

        bool interval() const {
            return variant == "C1+C2" || variant == "+interval";
        }

        bool zero_open() const {
            return variant == "C1+C2" || variant == "C1+zero";
        }

        bool amount() const {
            return variant == "C1+C2" || variant == "+amount";
        }
    };

    void compile_pattern_material(const InvestigationConfig &, MaterialPlan &);
    void compile_dc_material(const InvestigationConfig &, MaterialPlan &);
    MaterialPlan compile_investigation(const InvestigationConfig &);
    std::size_t named_mask(const MaterialPlan &, const std::string &);
    std::size_t named_comparison(const MaterialPlan &, const std::string &);
    std::size_t named_gate(const MaterialPlan &, const std::string &);
    std::vector<std::uint64_t> window_offsets(const InvestigationConfig &);
}
