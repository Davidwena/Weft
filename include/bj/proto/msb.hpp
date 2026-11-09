#pragma once
#include "bj/proto/ltbits.hpp"
#include "bj/common/bitvec.hpp"
#include "bj/common/ring.hpp"

namespace bj {
    template <class T> class MaskedRingOpen : public Step {
        unsigned k_;
        std::span<const T> x_, y_, r_, rp_;
        std::span<T> a_, b_;
        AlignedBuffer<T> send_a_, send_b_;
        RingSlot slot_a_{}, slot_b_{};
        bool two_, bound_ = false, used_ = false, done_ = false;

    public:
        MaskedRingOpen(unsigned k,
                       std::span<const T> x,
                       std::span<const T> r,
                       std::span<T> a,
                       std::span<const T> y = {},
                       std::span<const T> rp = {},
                       std::span<T> b = {},
                       bool two = false)
            : k_(k), x_(x), y_(y), r_(r), rp_(rp), a_(a), b_(b), send_a_(a.size()),
              send_b_(two ? a.size() : 0), two_(two) {
            if (k < 1 || k > 96 ||
                (k <= 32   ? 4u
                 : k <= 64 ? 8u
                           : 16u) != sizeof(T) ||
                x.size() != a.size() || r.size() != a.size() ||
                (two && (y.size() != a.size() || rp.size() != a.size() || b.size() != a.size()))) {
                throw std::invalid_argument("fresh comparison share shapes");
            }
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("masked ring step bound twice");
            }
            bound_ = true;
            slot_a_ = p.ring(k_, a_.size());
            if (two_) {
                slot_b_ = p.ring(k_, a_.size());
            }
        }

        void contribute(OpenWriter &w) override {
            if (!bound_ || used_) {
                throw std::logic_error("fresh masks already used");
            }
            used_ = true;
            const auto mask = native_mask<T>(k_);
#pragma omp parallel for if (a_.size() >= 16384) schedule(static)
            for (std::size_t i = 0; i < a_.size(); ++i) {
                send_a_.data()[i] = two_ ? (rp_[i] - x_[i]) & mask : (x_[i] + r_[i]) & mask;
                if (two_) {
                    send_b_.data()[i] = (y_[i] + r_[i]) & mask;
                }
            }
            w.ring<T>(slot_a_, send_a_.span());
            if (two_) {
                w.ring<T>(slot_b_, send_b_.span());
            }
        }

        void consume(OpenReader &r) override {
            if (!used_ || done_) {
                throw std::logic_error("fresh consume state");
            }
            done_ = true;
            r.ring<T>(slot_a_, a_);
            if (two_) {
                r.ring<T>(slot_b_, b_);
            }
        }
    };

    template <class T> class KnownMsbStep : public Step {
        unsigned k_, party_;
        std::size_t n_;
        std::span<const T> hat_;
        std::span<std::uint64_t> out_;
        BitVec sign_;
        std::unique_ptr<TableCompareStep<T>> lt_;
        bool bound_ = false, used_ = false, done_ = false;

    public:
        KnownMsbStep(unsigned k,
                     unsigned party,
                     std::size_t n,
                     std::span<const T> hat,
                     std::span<std::uint64_t> out,
                     std::shared_ptr<const BlockTable> table,
                     std::span<const std::byte> evaluation,
                     std::span<const std::byte> sign)
            : k_(k), party_(party), n_(n), hat_(hat), out_(out), sign_(n) {
            if (k < 1 || k > sizeof(T) * 8 ||
                (k <= 32   ? 4u
                 : k <= 64 ? 8u
                           : 16u) != sizeof(T) ||
                hat.size() != n || out.size() != bits_words(n)) {
                throw std::invalid_argument("MSB shape/native storage");
            }
            sign_.unpack(sign);
            if (k > 1) {
                lt_ = std::make_unique<TableCompareStep<T>>(
                    std::move(table), evaluation, party, hat, out);
            }
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("MSB bound twice");
            }
            bound_ = true;
            if (lt_) {
                lt_->bind(p);
            }
        }

        void contribute(OpenWriter &w) override {
            if (!bound_ || used_) {
                throw std::logic_error("MSB evaluation reused");
            }
            used_ = true;
            if (lt_) {
                lt_->contribute(w);
            }
        }

        void evaluate_local(std::span<std::uint64_t> out) const {
            if (lt_) {
                lt_->evaluate_local(out);
            } else {
                std::fill(out.begin(), out.end(), 0);
            }
            combine_sign(out);
        }

        void combine_sign(std::span<std::uint64_t> out) const {
#ifndef BJ_PROFILE_SERIAL
#pragma omp parallel for if (n_ >= 16384) schedule(static)
#endif
            for (std::size_t w = 0; w < out.size(); ++w) {
                std::uint64_t top = 0;
                if (party_ == 0) {
                    for (unsigned lane = 0; lane < 64 && w * 64 + lane < n_; ++lane) {
                        top |= std::uint64_t((hat_[w * 64 + lane] >> (k_ - 1)) & 1) << lane;
                    }
                }
                out[w] ^= sign_.words()[w] ^ top;
            }
        }

        void consume(OpenReader &r) override {
            if (!used_ || done_) {
                throw std::logic_error("MSB consume state");
            }
            done_ = true;
            if (lt_) {
                lt_->consume(r);
            }
            if (!lt_) {
                std::fill(out_.begin(), out_.end(), 0);
            }
            combine_sign(out_);
        }
    };

    struct KnownMsbMaterial {
        std::shared_ptr<const BlockTable> table;
        std::span<const std::byte> evaluation, sign;
    };

    inline KnownMsbMaterial load_msb_material(MaterialStore &s,
                                              const MaterialPlan &p,
                                              std::size_t c,
                                              unsigned e,
                                              std::shared_ptr<const BlockTable> table = {}) {
        s.claim_comparison(c, e);
        const auto &spec = p.comparisons.at(c);
        const auto &ids = s.layout().comparisons[c];
        const bool fresh = spec.kind == CompareKind::MsbFresh;
        if (spec.kind != CompareKind::MsbKnown && !fresh) {
            throw std::invalid_argument("expected MSB material");
        }
        if (spec.width > 1 && !table) {
            table = load_block_table(s, p, c);
        }
        return {std::move(table),
                spec.width > 1 ? s.packed(ids.evaluations[e][0]) : std::span<const std::byte>{},
                s.packed(ids.aux[fresh ? 1 : 0])};
    }

    template <class T> class MsbKnown : public KnownMsbStep<T> {
        using Base = KnownMsbStep<T>;

        MsbKnown(MaterialStore &s,
                 const MaterialPlan &p,
                 std::size_t c,
                 std::span<const T> hat,
                 std::span<std::uint64_t> out,
                 KnownMsbMaterial m)
            : Base(p.comparisons[c].width,
                   s.party(),
                   p.comparisons[c].count,
                   hat,
                   out,
                   std::move(m.table),
                   m.evaluation,
                   m.sign) {
        }

    public:
        MsbKnown(MaterialStore &s,
                 const MaterialPlan &p,
                 std::size_t c,
                 unsigned e,
                 std::span<const T> hat,
                 std::span<std::uint64_t> out,
                 std::shared_ptr<const BlockTable> table = {})
            : MsbKnown(s,
                       require_comparison(p, c, CompareKind::MsbKnown, p.comparisons.at(c).width),
                       c,
                       hat,
                       out,
                       load_msb_material(s, p, c, e, std::move(table))) {
        }
    };

    template <class T> class MsbFresh {
        AlignedBuffer<T> r_, hat_;
        std::unique_ptr<MaskedRingOpen<T>> opening_;
        std::unique_ptr<KnownMsbStep<T>> comparison_;

    public:
        MsbFresh(MaterialStore &s,
                 const MaterialPlan &p,
                 std::size_t c,
                 std::span<const T> x,
                 std::span<std::uint64_t> out)
            : r_(p.comparisons.at(c).count), hat_(r_.size()) {
            const auto &spec = p.comparisons[c];
            if (spec.kind != CompareKind::MsbFresh) {
                throw std::invalid_argument("expected fresh MSB");
            }
            const auto material = load_msb_material(s, p, c, 0);
            unpack_native<T>(spec.width, s.packed(s.layout().comparisons[c].aux[0]), r_.span());
            opening_ = std::make_unique<MaskedRingOpen<T>>(spec.width, x, r_.span(), hat_.span());
            comparison_ = std::make_unique<KnownMsbStep<T>>(spec.width,
                                                            s.party(),
                                                            spec.count,
                                                            hat_.span(),
                                                            out,
                                                            material.table,
                                                            material.evaluation,
                                                            material.sign);
        }

        Step &opening() noexcept {
            return *opening_;
        }

        Step &comparison() noexcept {
            return *comparison_;
        }

        void evaluate_local(std::span<std::uint64_t> out) const {
            comparison_->evaluate_local(out);
        }
    };

    template <unsigned K> class MsbKnownBatch : public MsbKnown<typename Ring<K>::T> {
        using T = typename Ring<K>::T;
        using Base = MsbKnown<T>;

    public:
        MsbKnownBatch(MaterialStore &s,
                      const MaterialPlan &p,
                      std::size_t c,
                      unsigned e,
                      std::span<const T> x,
                      std::span<std::uint64_t> out,
                      std::shared_ptr<const BlockTable> table = {})
            : Base(s,
                   require_comparison(p, c, CompareKind::MsbKnown, K),
                   c,
                   e,
                   x,
                   out,
                   std::move(table)) {
        }
    };

    template <unsigned K> class MsbFreshBatch : public MsbFresh<typename Ring<K>::T> {
        using T = typename Ring<K>::T;
        using Base = MsbFresh<T>;

    public:
        MsbFreshBatch(MaterialStore &s,
                      const MaterialPlan &p,
                      std::size_t c,
                      std::span<const T> x,
                      std::span<std::uint64_t> out)
            : Base(s, require_comparison(p, c, CompareKind::MsbFresh, K), c, x, out) {
        }
    };
}
