#pragma once
#include "bj/core/table.hpp"
#include "bj/ops/quicksort.hpp"
#include <memory>

namespace bj {
    class PatternOperator;

    struct DCStats {
        double seconds = 0, time_sort_seconds = 0, amount_sort_seconds = 0, levels_seconds = 0,
               shuffle_seconds = 0;
        std::uint64_t rounds = 0, hops = 0, contribution_bits = 0, shuffle_rounds = 0;
        std::array<SortStats, 3> sorts{};
        unsigned levels = 0;
        SortStats counterparty_sort{};
        std::array<SortStats, 2> restore_sorts{};
    };

    class DominanceCount {
        struct Impl;
        std::unique_ptr<Impl> impl_;
        std::unique_ptr<PatternOperator> pattern_;

    public:
        DominanceCount(const InvestigationConfig &,
                       MaterialStore &,
                       const MaterialPlan &,
                       const MaskedTable &,
                       const MaskedTable &,
                       std::span<const std::uint64_t>,
                       std::span<const std::uint64_t>);
        ~DominanceCount();
        void run(Net &);
        std::span<const std::uint64_t> count_shares() const;
        std::uint64_t overflow_share() const;
        std::span<const std::uint64_t> alarm_shares() const;
        const DCStats &stats() const;
    };
}
