#pragma once
#include "bj/core/comparison_dealer.hpp"

namespace bj {

    void make_mask_values(const MaterialPlan &,
                          std::vector<AlignedBuffer<std::byte>> &,
                          const ComparisonRandom &);
    void make_shuffle_values(const MaterialPlan &,
                             const MaterialLayout &,
                             MaterialMode,
                             std::span<const std::span<const std::byte>>,
                             const ComparisonEmitter &,
                             const ComparisonRandom &);
    void permutation_from_seed(std::span<const std::byte> seed, std::span<std::uint32_t> output);
}
