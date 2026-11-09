#include "bj/proto/compare_shape.hpp"
#include <bit>
#include <map>
#include <set>
#include <stdexcept>
#include <limits>

namespace bj {
    CompareShape::CompareShape(unsigned bits, unsigned block, bool eq)
        : ell(bits), d(block), equality(eq) {
        if (!ell || ell > 96 || !d || d > 9) {
            throw std::invalid_argument("comparison needs ell=1..96 and d=1..9");
        }
        blocks = (ell + d - 1) / d;
        if (blocks > 11) {
            throw std::invalid_argument("comparison expansion exceeds 11 blocks; increase d");
        }
        inputs = eq ? blocks : 2 * blocks - 1;
        if (eq) {
            BooleanGateShape g;
            for (unsigned b = 0; b < blocks; ++b) {
                g.inputs.push_back(b);
            }
            gates.push_back(std::move(g));
        } else {
            for (unsigned b = 0; b < blocks; ++b) {
                BooleanGateShape g;
                for (unsigned e = 0; e < b; ++e) {
                    g.inputs.push_back(e);
                }
                g.inputs.push_back(blocks - 1 + b);
                gates.push_back(std::move(g));
            }
        }
        std::set<std::uint32_t> unique;
        for (auto &gate : gates) {
            for (unsigned s = 0; s < (1u << gate.inputs.size()); ++s) {
                if (std::popcount(s) >= 2) {
                    std::uint32_t mask = 0;
                    for (unsigned j = 0; j < gate.inputs.size(); ++j) {
                        if (s & (1u << j)) {
                            mask |= 1u << gate.inputs[j];
                        }
                    }
                    unique.insert(mask);
                }
            }
        }
        products.assign(unique.begin(), unique.end());
        const auto expected =
            eq ? ((std::size_t{1} << blocks) - blocks - 1)
               : ((std::size_t{1} << (blocks - 1)) + (std::size_t{1} << blocks) - 2 * blocks - 1);
        if (products.size() != expected) {
            throw std::logic_error("unique product count disagrees with ltcost");
        }
        std::map<std::uint32_t, std::size_t> index;
        for (std::size_t i = 0; i < products.size(); ++i) {
            index[products[i]] = inputs + i;
        }
        for (auto &gate : gates) {
            gate.f_offset = f_bits;
            f_bits += std::size_t{1} << gate.inputs.size();
            gate.columns.resize(std::size_t{1} << gate.inputs.size(), SIZE_MAX);
            for (unsigned s = 1; s < gate.columns.size(); ++s) {
                std::uint32_t mask = 0;
                for (unsigned j = 0; j < gate.inputs.size(); ++j) {
                    if (s & (1u << j)) {
                        mask |= 1u << gate.inputs[j];
                    }
                }
                gate.columns[s] =
                    std::has_single_bit(mask) ? std::countr_zero(mask) : index.at(mask);
            }
        }
    }

    unsigned best_compare_d(unsigned ell, bool eq) {
        unsigned best = 0;
        std::size_t cost = SIZE_MAX;
        for (unsigned d = 1; d <= 9; ++d) {
            if ((ell + d - 1) / d <= 11) {
                CompareShape shape(ell, d, eq);
                if (shape.material_bits() < cost) {
                    cost = shape.material_bits();
                    best = d;
                }
            }
        }
        if (!best) {
            throw std::invalid_argument("invalid comparison width");
        }
        return best;
    }
}
