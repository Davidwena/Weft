#pragma once
#include "bj/ops/compact.hpp"
#include "bj/core/stage_stats.hpp"

namespace bj {

    MaterialPlan compile_batch_suspects(const InvestigationConfig &, std::size_t);
    MaterialPlan batch_suspect_view(const MaterialPlan &, std::size_t);

    class BatchSuspects {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        BatchSuspects(const InvestigationConfig &,
                      std::size_t,
                      MaterialStore &,
                      const MaterialPlan &,
                      const InputManager &);
        ~BatchSuspects();
        void prepare(Net &, unsigned);
        void run(Net &);
        std::vector<std::uint64_t> count_shares() const;
        std::vector<std::uint64_t> overflow_shares() const;
        const std::vector<StageStats> &stages() const;
        const std::vector<std::array<std::size_t, 2>> &slot_counts() const;
    };
}
