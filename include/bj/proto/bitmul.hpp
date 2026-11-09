#pragma once
#include "bj/core/round_exec.hpp"
#include "bj/core/table.hpp"

namespace bj {
    template <class T> class BitMul : public Step {
        unsigned party_, width_;
        std::size_t n_;
        std::span<const std::uint64_t> bit_;
        const MaskedColumn &value_;
        std::span<T> out_;
        BitVec tau_, hat_;
        AlignedBuffer<T> arithmetic_, product_;
        BitSlot slot_;
        bool bound_ = false, used_ = false, done_ = false;

    public:
        BitMul(MaterialStore &s,
               const MaterialPlan &p,
               std::size_t id,
               std::span<const std::uint64_t> bit,
               const MaskedColumn &x,
               std::span<T> out)
            : party_(s.party()), width_(p.bitmuls.at(id).width), n_(out.size()), bit_(bit),
              value_(x), out_(out), tau_(n_), hat_(n_), arithmetic_(n_), product_(n_) {
            const auto &spec = p.bitmuls[id];
            if (spec.count != n_ || x.count != n_ || x.width < width_ ||
                bit.size() != bits_words(n_) ||
                sizeof(T) != (width_ <= 32   ? 4u
                              : width_ <= 64 ? 8u
                                             : 16u)) {
                throw std::invalid_argument("bitmul shape/native type");
            }
            s.claim_bitmul(id);
            const auto &ids = s.layout().bitmuls[id];
            tau_.unpack(s.packed(ids[0]));
            unpack_native<T>(width_, s.packed(ids[1]), arithmetic_.span());
            unpack_native<T>(width_, s.packed(ids[2]), product_.span());
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("bitmul bound twice");
            }
            bound_ = true;
            slot_ = p.bits(n_);
        }

        void contribute(OpenWriter &w) override {
            if (!bound_ || used_ || !value_.dealer_knows_mask) {
                throw std::logic_error("bitmul requires fresh material and dealer-known mask");
            }
            used_ = true;
            for (std::size_t i = 0; i < hat_.words().size(); ++i) {
                hat_.words()[i] = bit_[i] ^ tau_.words()[i];
            }
            w.bits(slot_, hat_.words());
        }

        void consume(OpenReader &r) override {
            if (!used_ || done_) {
                throw std::logic_error("bitmul consume state");
            }
            done_ = true;
            r.bits(slot_, hat_.words());
            const auto reduce = native_mask<T>(width_);
#pragma omp parallel for if (n_ >= 16384) schedule(static)
            for (std::size_t i = 0; i < n_; ++i) {
                const auto d = static_cast<T>(value_.width <= 32   ? uint128_t(value_.d32.data()[i])
                                              : value_.width <= 64 ? uint128_t(value_.d64.data()[i])
                                                                   : value_.d128.data()[i]);
                const auto m = static_cast<T>(value_.width <= 32   ? uint128_t(value_.m32.data()[i])
                                              : value_.width <= 64 ? uint128_t(value_.m64.data()[i])
                                                                   : value_.m128.data()[i]);
                const T b = hat_.get(i);
                out_[i] = (b * ((party_ == 0 ? d : T{0}) - m) +
                           (T{1} - 2 * b) * (d * arithmetic_.data()[i] - product_.data()[i])) &
                          reduce;
            }
        }
    };

    inline std::size_t named_bitmul(const MaterialPlan &p, const std::string &name) {
        for (std::size_t i = 0; i < p.bitmuls.size(); ++i) {
            if (p.bitmuls[i].name == name) {
                return i;
            }
        }
        throw std::invalid_argument("unknown bitmul " + name);
    }
}
