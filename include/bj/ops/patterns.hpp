#pragma once
#include "bj/ops/dc.hpp"
#include "bj/ops/nl.hpp"

namespace bj {

    class PatternOperator {
    public:
        virtual ~PatternOperator() = default;
        virtual void run(Net &) = 0;
        virtual std::span<const std::uint64_t> count_shares() const = 0;
        virtual std::uint64_t overflow_share() const = 0;
        virtual std::span<const std::uint64_t> alarm_shares() const = 0;
        virtual const DCStats &stats() const = 0;
    };

    std::unique_ptr<PatternOperator>
    make_pattern_operator(const InvestigationConfig &,
                          MaterialStore &,
                          const MaterialPlan &,
                          const MaskedTable &,
                          const MaskedTable &,
                          std::span<const std::uint64_t>,
                          std::span<const std::uint64_t>,
                          std::span<const std::uint64_t> extra = {});
}
