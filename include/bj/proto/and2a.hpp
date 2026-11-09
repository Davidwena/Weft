#pragma once
#include "bj/core/material.hpp"
#include "bj/core/round_exec.hpp"
#include "bj/common/ring.hpp"
#include <array>

namespace bj {

    template <unsigned M, class T>
#ifdef BJ_PROFILE_SERIAL
    [[gnu::noinline]]
#endif
    void evaluate_and2a(std::span<const T> table,
                        const std::array<std::span<const std::uint64_t>, M> &opened,
                        std::span<T> out) {
        static_assert(M >= 1 && M <= 4);
        const auto n = out.size();
        if (table.size() != n * (1u << M)) {
            throw std::invalid_argument("AND2A table size");
        }
        for (auto col : opened) {
            if (col.size() != bits_words(n)) {
                throw std::invalid_argument("AND2A opened column size");
            }
        }
#ifndef BJ_PROFILE_SERIAL
#pragma omp parallel for if (n >= 16384) schedule(static)
#endif
        for (std::size_t i = 0; i < n; ++i) {
            unsigned index = 0;
            for (unsigned j = 0; j < M; ++j) {
                index |= static_cast<unsigned>((opened[j][i / 64] >> (i % 64)) & 1) << j;
            }
            out[i] = table[index * n + i];
        }
    }

    template <unsigned M, class T> class FusedAnd2A : public Step {
        static_assert(M >= 1 && M <= 4 && (std::is_unsigned_v<T> || std::is_same_v<T, uint128_t>) &&
                      (sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16));
        unsigned width_, party_;
        std::size_t n_, words_;
        std::array<std::span<const std::uint64_t>, M> inputs_;
        std::array<BitSlot, M> slots_{};
        AlignedBuffer<std::uint64_t> tau_, masked_;
        AlignedBuffer<T> table_;
        std::span<T> output_;
        bool bound_ = false, contributed_ = false, consumed_ = false;

    public:
        FusedAnd2A(MaterialStore &material,
                   const MaterialPlan &plan,
                   std::size_t gate,
                   const std::array<std::span<const std::uint64_t>, M> &inputs,
                   std::span<T> output)
            : width_(plan.gates.at(gate).width), party_(material.party()),
              n_(plan.gates.at(gate).count), words_(bits_words(n_)), inputs_(inputs),
              tau_(M * words_), masked_(M * words_), table_(n_ * (1u << M)), output_(output) {
            const auto &spec = plan.gates[gate];
            if (spec.m != M || output.size() != n_ ||
                (width_ <= 32   ? 4u
                 : width_ <= 64 ? 8u
                                : 16u) != sizeof(T) ||
                width_ > sizeof(T) * 8) {
                throw std::invalid_argument("AND2A native ring/shape mismatch");
            }
            for (auto in : inputs) {
                if (in.size() != words_) {
                    throw std::invalid_argument("AND2A input size");
                }
            }
            material.claim_gate(gate);
            const auto &layout = material.layout();
            const auto masks = material.packed(layout.gate_masks[gate]);
            for (unsigned j = 0; j < M; ++j) {
                load_bits(masks, j * n_, n_, tau_.span().subspan(j * words_, words_));
            }
            auto empty = table_.span().first(n_);
            std::fill(empty.begin(), empty.end(), party_ == 0 ? T{1} : T{0});
            for (unsigned subset = 1; subset < (1u << M); ++subset) {
                unpack_native<T>(width_,
                                 material.packed(layout.gate_products[gate][subset - 1]),
                                 table_.span().subspan(subset * n_, n_));
            }
            const auto mask = native_mask<T>(width_);

#pragma omp parallel for if (n_ >= 16384) schedule(static)
            for (std::size_t i = 0; i < n_; ++i) {
                std::array<T, 1u << M> a{};
                for (unsigned s = 0; s < (1u << M); ++s) {
                    a[s] = table_.data()[s * n_ + i];
                }
                for (unsigned bit = 0; bit < M; ++bit) {
                    for (unsigned s = 0; s < (1u << M); ++s) {
                        if (!(s & (1u << bit))) {
                            const T low = a[s], high = a[s | (1u << bit)];
                            a[s] = high;
                            a[s | (1u << bit)] = low - high;
                        }
                    }
                }
                for (unsigned h = 0; h < (1u << M); ++h) {
                    table_.data()[h * n_ + i] = a[h] & mask;
                }
            }
        }

        void bind(RoundPlan &plan) override {
            if (bound_) {
                throw std::logic_error("AND2A batch already bound");
            }
            bound_ = true;
            for (auto &slot : slots_) {
                slot = plan.bits(n_);
            }
        }

        void contribute(OpenWriter &writer) override {
            if (!bound_ || contributed_) {
                throw std::logic_error("AND2A masks may only be opened once");
            }
            contributed_ = true;
#pragma omp parallel for if (M * words_ >= 16384) schedule(static)
            for (std::size_t i = 0; i < M * words_; ++i) {
                masked_.data()[i] = inputs_[i / words_][i % words_] ^ tau_.data()[i];
            }
            for (unsigned j = 0; j < M; ++j) {
                writer.bits(slots_[j], masked_.span().subspan(j * words_, words_));
            }
        }

        void consume(OpenReader &reader) override {
            if (!contributed_ || consumed_) {
                throw std::logic_error("AND2A consume state");
            }
            consumed_ = true;
            std::array<std::span<const std::uint64_t>, M> hats;
            for (unsigned j = 0; j < M; ++j) {
                auto col = masked_.span().subspan(j * words_, words_);
                reader.bits(slots_[j], col);
                hats[j] = col;
            }
            evaluate_and2a<M, T>(table_.span(), hats, output_);
        }

        void evaluate_local() {
            if (!consumed_) {
                throw std::logic_error("local evaluation requires a completed gate");
            }
            std::array<std::span<const std::uint64_t>, M> hats;
            for (unsigned j = 0; j < M; ++j) {
                hats[j] = masked_.span().subspan(j * words_, words_);
            }
            evaluate_and2a<M, T>(table_.span(), hats, output_);
        }

        std::span<const T> local_table() const noexcept {
            return table_.span();
        }

        std::span<const std::uint64_t> opened_column(unsigned j) const {
            if (j >= M) {
                throw std::out_of_range("AND2A column");
            }
            return masked_.span().subspan(j * words_, words_);
        }
    };

    template <unsigned M, unsigned K> class And2ABatch : public FusedAnd2A<M, typename Ring<K>::T> {
        static_assert(M >= 2 && M <= 4);
        using Base = FusedAnd2A<M, typename Ring<K>::T>;

        static const MaterialPlan &checked(const MaterialPlan &p, std::size_t g) {
            if (p.gates.at(g).width != K) {
                throw std::invalid_argument("typed AND2A ring width mismatch");
            }
            return p;
        }

    public:
        And2ABatch(MaterialStore &material,
                   const MaterialPlan &p,
                   std::size_t gate,
                   const std::array<std::span<const std::uint64_t>, M> &inputs,
                   std::span<typename Ring<K>::T> output)
            : Base(material, checked(p, gate), gate, inputs, output) {
        }
    };
}
