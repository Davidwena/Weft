#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bj {
    struct BooleanGateShape {
        std::vector<unsigned> inputs;
        std::vector<std::size_t> columns;
        std::size_t f_offset = 0;
    };

    struct CompareShape {
        unsigned ell = 0, d = 0, blocks = 0, inputs = 0;
        bool equality = false;
        std::vector<std::uint32_t> products;
        std::vector<BooleanGateShape> gates;
        std::size_t f_bits = 0;
        CompareShape(unsigned ell, unsigned d, bool equality = false);

        std::size_t table_bits() const noexcept {
            return std::size_t(blocks) * (std::size_t{1} << d);
        }

        std::size_t eval_bits() const noexcept {
            return inputs + products.size();
        }

        std::size_t material_bits() const noexcept {
            return table_bits() + eval_bits();
        }
    };

    unsigned best_compare_d(unsigned ell, bool equality = false);
}
