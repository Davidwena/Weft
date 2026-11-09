#pragma once
#include "bj/proto/and2a.hpp"

namespace bj {

    template <unsigned K> class B2ABatch : public FusedAnd2A<1, typename Ring<K>::T> {
        using Base = FusedAnd2A<1, typename Ring<K>::T>;

        static const MaterialPlan &checked(const MaterialPlan &p, std::size_t g) {
            if (p.gates.at(g).width != K) {
                throw std::invalid_argument("typed B2A ring width mismatch");
            }
            return p;
        }

    public:
        B2ABatch(MaterialStore &material,
                 const MaterialPlan &p,
                 std::size_t gate,
                 const std::array<std::span<const std::uint64_t>, 1> &inputs,
                 std::span<typename Ring<K>::T> output)
            : Base(material, checked(p, gate), gate, inputs, output) {
        }
    };
}
