#pragma once
#include "bj/proto/interval.hpp"
#include "bj/proto/and2a.hpp"
#include "bj/core/table.hpp"

namespace bj {
    class BandPair {
    public:
        virtual ~BandPair() = default;
        virtual void prepare() = 0;
        virtual std::span<Step *const> opening_steps() = 0;
        virtual std::span<Step *const> comparison_steps() = 0;
        virtual std::span<Step *const> product_steps() = 0;
        virtual std::span<const std::uint64_t> shares(std::size_t window) const = 0;
        virtual void combine() = 0;
        virtual void evaluate_local() = 0;
    };

    std::unique_ptr<BandPair> make_band_pair(const InvestigationConfig &,
                                             MaterialStore &,
                                             const MaterialPlan &,
                                             const MaskedTable &inbound,
                                             const MaskedTable &outbound,
                                             std::span<const std::uint64_t> extra = {});
}
