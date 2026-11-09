#pragma once
#include "bj/core/material.hpp"
#include "bj/core/round_exec.hpp"

namespace bj {
    template <class T> class BeaverProduct : public Step {
        unsigned party_;
        std::span<const T> x_, y_;
        std::span<T> out_;
        AlignedBuffer<T> a_, b_, c_, d_, e_;
        RingSlot ds_, es_;
        bool bound_ = false, used_ = false, done_ = false;

    public:
        BeaverProduct(MaterialStore &s,
                      const MaterialPlan &p,
                      std::size_t index,
                      std::span<const T> x,
                      std::span<const T> y,
                      std::span<T> out)
            : party_(s.party()), x_(x), y_(y), out_(out), a_(out.size()), b_(out.size()),
              c_(out.size()), d_(out.size()), e_(out.size()) {
            const auto &m = p.multiplications.at(index);
            if (m.width != sizeof(T) * 8 || m.count != out.size() || x.size() != out.size() ||
                y.size() != out.size()) {
                throw std::invalid_argument("triple shape");
            }
            s.claim_multiply(index);
            const auto &ids = s.layout().multiplications.at(index);
            unpack_native<T>(m.width, s.packed(ids[0]), a_.span());
            unpack_native<T>(m.width, s.packed(ids[1]), b_.span());
            unpack_native<T>(m.width, s.packed(ids[2]), c_.span());
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("triple bound twice");
            }
            bound_ = true;
            ds_ = p.ring(sizeof(T) * 8, out_.size());
            es_ = p.ring(sizeof(T) * 8, out_.size());
        }

        void contribute(OpenWriter &w) override {
            if (!bound_ || used_) {
                throw std::logic_error("triple reused");
            }
            used_ = true;
#pragma omp parallel for if (out_.size() >= 16384) schedule(static)
            for (std::size_t i = 0; i < out_.size(); ++i) {
                d_.data()[i] = x_[i] - a_.data()[i];
                e_.data()[i] = y_[i] - b_.data()[i];
            }
            w.ring<T>(ds_, d_.span());
            w.ring<T>(es_, e_.span());
        }

        void consume(OpenReader &r) override {
            if (!used_ || done_) {
                throw std::logic_error("triple consume");
            }
            done_ = true;
            r.ring<T>(ds_, d_.span());
            r.ring<T>(es_, e_.span());
#pragma omp parallel for if (out_.size() >= 16384) schedule(static)
            for (std::size_t i = 0; i < out_.size(); ++i) {
                out_[i] = c_.data()[i] + d_.data()[i] * b_.data()[i] + e_.data()[i] * a_.data()[i] +
                          (party_ == 0 ? d_.data()[i] * e_.data()[i] : T{0});
            }
        }
    };
}
