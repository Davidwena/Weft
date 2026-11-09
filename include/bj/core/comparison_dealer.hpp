#pragma once
#include "bj/core/material.hpp"
#include "bj/proto/compare_shape.hpp"
#include <functional>

namespace bj {

    using ComparisonEmitter = std::function<void(std::size_t, std::span<const std::byte>)>;
    using ComparisonRandom = std::function<void(std::span<std::byte>)>;
    void make_comparison_evaluation(const CompareShape &,
                                    std::size_t count,
                                    std::span<std::byte> output,
                                    const ComparisonRandom &);

    template <class T>
    void make_comparison_table(const CompareShape &shape,
                               std::span<const T> thresholds,
                               std::span<std::byte> output) {
        const auto n = thresholds.size(), entries = std::size_t{1} << shape.d;
        if (output.size() != bits_bytes(shape.table_bits() * n)) {
            throw std::invalid_argument("one-hot table size");
        }
        std::fill(output.begin(), output.end(), std::byte{0});
        for (unsigned b = 0; b < shape.blocks; ++b) {
            for (std::size_t i = 0; i < n; ++i) {
                const auto value =
                    (thresholds[i] >> ((shape.blocks - 1 - b) * shape.d)) & (entries - 1);
                const auto bit = (b * n + i) * entries + value;
                output[bit / 8] |= std::byte(1u << (bit % 8));
            }
        }
    }

    void generate_comparison_material(const MaterialPlan &,
                                      const MaterialLayout &,
                                      std::span<const std::span<const std::byte>> input_masks,
                                      const ComparisonEmitter &,
                                      const ComparisonRandom &);
}
