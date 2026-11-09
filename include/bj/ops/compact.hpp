#pragma once
#include "bj/core/input.hpp"
#include "bj/core/table.hpp"
#include "bj/common/timer.hpp"

namespace bj {
    struct CompactStats {
        std::array<double, 8> seconds{};
        std::uint64_t open_rounds = 0, hops = 0, contribution_bits = 0, logical_rounds = 0;
        std::size_t shuffle_bytes = 0;
    };

    MaterialPlan compile_candidate_compact(const InvestigationConfig &);

    class Compact {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        Compact(const InvestigationConfig &,
                MaterialStore &,
                const MaterialPlan &,
                const InputManager &,
                std::span<const std::uint64_t> row_union = {},
                std::array<std::uint32_t, 2> prior_counts = {});

        Compact(const InvestigationConfig &,
                MaterialStore &,
                const MaterialPlan &,
                const MaskedTable &candidates,
                std::size_t incoming_rows);
        ~Compact();
        void run(Net &);
        void evaluate_local();
        const MaskedTable &inbound() const;
        const MaskedTable &outbound() const;
        std::span<const std::uint64_t> omega_in() const;
        std::span<const std::uint64_t> omega_out() const;
        const CompactStats &stats() const;
        std::array<std::uint32_t, 2> cumulative_count_shares() const;
    };
}
