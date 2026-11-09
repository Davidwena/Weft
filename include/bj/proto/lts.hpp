#pragma once
#include "bj/proto/msb.hpp"

namespace bj {
    struct LtsMaterial {
        unsigned width = 0, party = 0;
        std::size_t count = 0;
        bool stored = false;
        std::array<std::shared_ptr<const BlockTable>, 3> tables;
        std::array<std::span<const std::byte>, 3> evaluations;
        std::span<const std::byte> sigma;
    };

    inline LtsMaterial
    load_lts_material(MaterialStore &store, const MaterialPlan &p, std::size_t c) {
        const auto &spec = p.comparisons.at(c);
        const auto &ids = store.layout().comparisons.at(c);
        if (spec.kind != CompareKind::LtsKnown && spec.kind != CompareKind::LtsFresh) {
            throw std::invalid_argument("expected LTS material");
        }
        store.claim_comparison(c, 0);
        LtsMaterial m;
        m.width = spec.width;
        m.party = store.party();
        m.count = spec.count;
        m.stored = spec.kind == CompareKind::LtsKnown;
        for (unsigned t = 0; t < 3; ++t) {
            m.tables[t] = load_block_table(store, p, c, t);
            m.evaluations[t] = store.packed(ids.evaluations[0][t]);
        }
        m.sigma = store.packed(ids.aux.back());
        return m;
    }

    template <class T> class LtsCompareStep : public Step {
        unsigned k_, party_;
        bool stored_, bound_ = false, used_ = false, done_ = false;
        std::span<const T> x_, y_;
        std::span<T> a_, b_;
        AlignedBuffer<T> Y_;
        std::span<std::uint64_t> out_;
        BitVec sigma_, alpha_, beta_, delta_, gamma_;
        std::unique_ptr<TableCompareStep<T>> first_, second_, third_;

    public:
        LtsCompareStep(const LtsMaterial &m,
                       std::span<const T> x,
                       std::span<const T> y,
                       std::span<T> a,
                       std::span<T> b,
                       std::span<std::uint64_t> out)
            : k_(m.width), party_(m.party), stored_(m.stored), x_(x), y_(y), a_(a), b_(b),
              Y_(a.size()), out_(out), sigma_(a.size()), alpha_(a.size()), beta_(a.size()),
              delta_(a.size()), gamma_(a.size()) {
            if ((k_ <= 32   ? 4u
                 : k_ <= 64 ? 8u
                            : 16u) != sizeof(T) ||
                x.size() != m.count || y.size() != m.count || a.size() != m.count ||
                b.size() != m.count || out.size() != bits_words(m.count)) {
                throw std::invalid_argument("LTS batch shape/native width");
            }
            sigma_.unpack(m.sigma);
            first_ = std::make_unique<TableCompareStep<T>>(
                m.tables[0], m.evaluations[0], party_, b_, alpha_.words(), true);
            second_ = std::make_unique<TableCompareStep<T>>(
                m.tables[1], m.evaluations[1], party_, a_, beta_.words());
            third_ = std::make_unique<TableCompareStep<T>>(
                m.tables[2], m.evaluations[2], party_, Y_.span(), delta_.words());
        }

        LtsCompareStep(MaterialStore &store,
                       const MaterialPlan &p,
                       std::size_t c,
                       std::span<const T> x,
                       std::span<const T> y,
                       std::span<T> a,
                       std::span<T> b,
                       std::span<std::uint64_t> out)
            : LtsCompareStep(load_lts_material(store, p, c), x, y, a, b, out) {
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("LTS already bound");
            }
            bound_ = true;
            first_->bind(p);
            second_->bind(p);
            third_->bind(p);
        }

        void contribute(OpenWriter &w) override {
            if (!bound_ || used_) {
                throw std::logic_error("LTS masks reused");
            }
            used_ = true;
            const auto mask = native_mask<T>(k_);
#pragma omp parallel for if (a_.size() >= 16384) schedule(static)
            for (std::size_t i = 0; i < a_.size(); ++i) {
                if (stored_) {
                    a_[i] = (T{0} - x_[i]) & mask;
                    b_[i] = y_[i] & mask;
                }
                Y_.data()[i] = (a_[i] + b_[i]) & mask;
            }
            for (std::size_t word = 0; word < gamma_.words().size(); ++word) {
                std::uint64_t bits = 0;
                for (unsigned j = 0; j < 64 && word * 64 + j < a_.size(); ++j) {
                    bits |= std::uint64_t(Y_.data()[word * 64 + j] < b_[word * 64 + j]) << j;
                }
                gamma_.words()[word] = bits;
            }
            first_->contribute(w);
            second_->contribute(w);
            third_->contribute(w);
        }

        void evaluate_local(std::span<std::uint64_t> out) {
            first_->evaluate_local(alpha_.words());
            second_->evaluate_local(beta_.words());
            third_->evaluate_local(delta_.words());
            combine(out);
        }

        void combine(std::span<std::uint64_t> out) const {
#ifndef BJ_PROFILE_SERIAL
#pragma omp parallel for if (a_.size() >= 16384) schedule(static)
#endif
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = alpha_.words()[i] ^ beta_.words()[i] ^ delta_.words()[i] ^
                         sigma_.words()[i] ^ (party_ == 0 ? gamma_.words()[i] : 0);
            }
        }

        void consume(OpenReader &r) override {
            if (!used_ || done_) {
                throw std::logic_error("LTS consume state");
            }
            done_ = true;
            first_->consume(r);
            second_->consume(r);
            third_->consume(r);
            combine(out_);
        }
    };

    template <class T> class LtsBatch {
        AlignedBuffer<T> a_, b_, r_, rp_;
        std::unique_ptr<MaskedRingOpen<T>> opening_;
        std::unique_ptr<LtsCompareStep<T>> comparison_;

    public:
        LtsBatch(MaterialStore &store,
                 const MaterialPlan &p,
                 std::size_t c,
                 std::span<const T> x,
                 std::span<const T> y,
                 std::span<std::uint64_t> out)
            : a_(p.comparisons.at(c).count), b_(a_.size()),
              r_(p.comparisons[c].kind == CompareKind::LtsFresh ? a_.size() : 0), rp_(r_.size()) {
            const auto &spec = p.comparisons[c];
            const auto &ids = store.layout().comparisons[c];
            if (spec.kind == CompareKind::LtsFresh) {
                unpack_native<T>(spec.width, store.packed(ids.aux[0]), r_.span());
                unpack_native<T>(spec.width, store.packed(ids.aux[1]), rp_.span());
                opening_ = std::make_unique<MaskedRingOpen<T>>(
                    spec.width, x, r_.span(), a_.span(), y, rp_.span(), b_.span(), true);
            }
            comparison_ =
                std::make_unique<LtsCompareStep<T>>(store, p, c, x, y, a_.span(), b_.span(), out);
        }

        Step *opening() noexcept {
            return opening_.get();
        }

        Step &comparison() noexcept {
            return *comparison_;
        }

        void evaluate_local(std::span<std::uint64_t> out) const {
            comparison_->evaluate_local(out);
        }
    };

    template <unsigned K> class LtsKnownBatch : public LtsBatch<typename Ring<K>::T> {
        using T = typename Ring<K>::T;
        using Base = LtsBatch<T>;

    public:
        LtsKnownBatch(MaterialStore &s,
                      const MaterialPlan &p,
                      std::size_t c,
                      std::span<const T> x,
                      std::span<const T> y,
                      std::span<std::uint64_t> out)
            : Base(s, require_comparison(p, c, CompareKind::LtsKnown, K), c, x, y, out) {
        }
    };

    template <unsigned K> class LtsFreshBatch : public LtsBatch<typename Ring<K>::T> {
        using T = typename Ring<K>::T;
        using Base = LtsBatch<T>;

    public:
        LtsFreshBatch(MaterialStore &s,
                      const MaterialPlan &p,
                      std::size_t c,
                      std::span<const T> x,
                      std::span<const T> y,
                      std::span<std::uint64_t> out)
            : Base(s, require_comparison(p, c, CompareKind::LtsFresh, K), c, x, y, out) {
        }
    };
}
