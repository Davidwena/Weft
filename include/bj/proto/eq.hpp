#pragma once
#include "bj/proto/msb.hpp"
#include "bj/common/ring.hpp"

namespace bj {
    template <class T> class EqComparison : public TableCompareStep<T> {
        using Base = TableCompareStep<T>;

        static std::span<const std::byte>
        material(MaterialStore &s, const MaterialPlan &p, std::size_t c, unsigned e) {
            if (p.comparisons.at(c).kind != CompareKind::Eq &&
                p.comparisons.at(c).kind != CompareKind::EqFresh) {
                throw std::invalid_argument("expected EQ material");
            }
            s.claim_comparison(c, e);
            return s.packed(s.layout().comparisons[c].evaluations.at(e)[0]);
        }

    public:
        EqComparison(MaterialStore &s,
                     const MaterialPlan &p,
                     std::size_t c,
                     unsigned e,
                     std::span<const T> hat,
                     std::span<std::uint64_t> out,
                     std::shared_ptr<const BlockTable> table = {})
            : Base(table ? table : load_block_table(s, p, c),
                   material(s, p, c, e),
                   s.party(),
                   hat,
                   out) {
        }
    };

    template <class T> class EqFresh {
        AlignedBuffer<T> mask_, hat_;
        std::unique_ptr<MaskedRingOpen<T>> opening_;
        std::unique_ptr<EqComparison<T>> comparison_;

    public:
        EqFresh(MaterialStore &s,
                const MaterialPlan &p,
                std::size_t c,
                std::span<const T> x,
                std::span<std::uint64_t> out)
            : mask_(p.comparisons.at(c).count), hat_(mask_.size()) {
            const auto &spec = p.comparisons[c];
            if (spec.kind != CompareKind::EqFresh || spec.evaluations != 1) {
                throw std::invalid_argument("expected fresh EQ");
            }
            unpack_native<T>(spec.width, s.packed(s.layout().comparisons[c].aux[0]), mask_.span());
            opening_ =
                std::make_unique<MaskedRingOpen<T>>(spec.width, x, mask_.span(), hat_.span());
            comparison_ = std::make_unique<EqComparison<T>>(s, p, c, 0, hat_.span(), out);
        }

        Step &opening() {
            return *opening_;
        }

        Step &comparison() {
            return *comparison_;
        }
    };

    template <unsigned K> class EqBatch : public EqComparison<typename Ring<K>::T> {
        using T = typename Ring<K>::T;
        using Base = EqComparison<T>;

    public:
        EqBatch(MaterialStore &s,
                const MaterialPlan &p,
                std::size_t c,
                unsigned e,
                std::span<const T> x,
                std::span<std::uint64_t> out,
                std::shared_ptr<const BlockTable> table = {})
            : Base(
                  s, require_comparison(p, c, CompareKind::Eq, K), c, e, x, out, std::move(table)) {
        }
    };
}
