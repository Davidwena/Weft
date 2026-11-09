#pragma once
#include "bj/common/aligned.hpp"
#include <algorithm>
#include <cstdint>

namespace bj {
    class BitVec {
        std::size_t bits_;
        AlignedBuffer<std::uint64_t> words_;

        void trim() noexcept {
            if (bits_ % 64) {
                words_.data()[bits_ / 64] &= (std::uint64_t{1} << (bits_ % 64)) - 1;
            }
        }

    public:
        explicit BitVec(std::size_t bits) : bits_(bits), words_(bits / 64 + (bits % 64 != 0)) {
            clear();
        }

        std::size_t size() const noexcept {
            return bits_;
        }

        std::span<std::uint64_t> words() noexcept {
            return words_.span();
        }

        std::span<const std::uint64_t> words() const noexcept {
            return words_.span();
        }

        void clear() noexcept {
            std::fill(words_.span().begin(), words_.span().end(), 0);
        }

        bool get(std::size_t i) const {
            if (i >= bits_) {
                throw std::out_of_range("bit index");
            }
            return (words_.data()[i / 64] >> (i % 64)) & 1;
        }

        void set(std::size_t i, bool b) {
            if (i >= bits_) {
                throw std::out_of_range("bit index");
            }
            auto &w = words_.data()[i / 64];
            auto mask = std::uint64_t{1} << (i % 64);
            w = (w & ~mask) | (std::uint64_t(b) << (i % 64));
        }

        static void xor_into(const BitVec &a, const BitVec &b, BitVec &out) {
            combine<false>(a, b, out);
        }

        static void and_into(const BitVec &a, const BitVec &b, BitVec &out) {
            combine<true>(a, b, out);
        }

        template <bool And> static void combine(const BitVec &a, const BitVec &b, BitVec &out) {
            if (a.size() != b.size() || a.size() != out.size()) {
                throw std::invalid_argument("bitvec size");
            }
            const auto n = a.words_.size();
#pragma omp parallel for if (n >= 16384) schedule(static)
            for (std::size_t i = 0; i < n; ++i) {
                if constexpr (And) {
                    out.words_.data()[i] = a.words_.data()[i] & b.words_.data()[i];
                } else {
                    out.words_.data()[i] = a.words_.data()[i] ^ b.words_.data()[i];
                }
            }
            out.trim();
        }

        std::size_t packed_bytes() const noexcept {
            return bits_ / 8 + (bits_ % 8 != 0);
        }

        void pack(std::span<std::byte> out) const {
            if (out.size() != packed_bytes()) {
                throw std::invalid_argument("bit pack size");
            }
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = std::byte(words_.data()[i / 8] >> ((i % 8) * 8));
            }
            if (bits_ % 8) {
                out.back() &= std::byte((1u << (bits_ % 8)) - 1);
            }
        }

        void unpack(std::span<const std::byte> in) {
            if (in.size() != packed_bytes()) {
                throw std::invalid_argument("bit unpack size");
            }
            clear();
            for (std::size_t i = 0; i < in.size(); ++i) {
                words_.data()[i / 8] |= std::uint64_t(std::to_integer<unsigned>(in[i]))
                                        << ((i % 8) * 8);
            }
            trim();
        }
    };
}
