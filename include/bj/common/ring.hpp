#pragma once
#include "bj/common/packed.hpp"
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace bj {
    template <unsigned K> struct Ring {
        static_assert(K >= 1 && K <= 96);
        using T = std::conditional_t<(K <= 32),
                                     std::uint32_t,
                                     std::conditional_t<(K <= 64), std::uint64_t, uint128_t>>;

        static constexpr T mask() noexcept {
            return native_mask<T>(K);
        }

        static constexpr T normalize(T x) noexcept {
            return x & mask();
        }

        static constexpr bool msb(T x) noexcept {
            return (x >> (K - 1)) & 1;
        }

        static constexpr T add(T x, T y) noexcept {
            return normalize(x + y);
        }

        static constexpr T sub(T x, T y) noexcept {
            return normalize(x - y);
        }

        static constexpr T mul(T x, T y) noexcept {
            return normalize(x * y);
        }

        template <unsigned K2> static constexpr typename Ring<K2>::T reduce(T x) noexcept {
            static_assert(K2 <= K);
            return Ring<K2>::normalize(static_cast<typename Ring<K2>::T>(x));
        }
    };

    template <unsigned K> constexpr std::size_t packed_ring_bytes(std::size_t n) {
        if (n > (std::numeric_limits<std::size_t>::max() - 7) / K) {
            throw std::length_error("packed size overflow");
        }
        return (n * K + 7) / 8;
    }

    template <unsigned K>
    void pack_ring(std::span<const typename Ring<K>::T> in, std::span<std::byte> out) {
        if (out.size() != packed_ring_bytes<K>(in.size())) {
            throw std::invalid_argument("pack size");
        }
        for (auto &b : out) {
            b = std::byte{0};
        }
        std::size_t bit = 0;
        for (auto x : in) {
            x = Ring<K>::normalize(x);
            for (unsigned used = 0; used < K;) {
                const unsigned shift = static_cast<unsigned>(bit % 8);
                const unsigned take = std::min(8 - shift, K - used);
                out[bit / 8] |=
                    std::byte(static_cast<unsigned>((x >> used) & ((1u << take) - 1)) << shift);
                bit += take;
                used += take;
            }
        }
    }

    template <unsigned K>
    void unpack_ring(std::span<const std::byte> in, std::span<typename Ring<K>::T> out) {
        if (in.size() != packed_ring_bytes<K>(out.size())) {
            throw std::invalid_argument("unpack size");
        }
        std::size_t bit = 0;
        for (auto &x : out) {
            x = 0;
            for (unsigned used = 0; used < K;) {
                const unsigned shift = static_cast<unsigned>(bit % 8);
                const unsigned take = std::min(8 - shift, K - used);
                x |= static_cast<typename Ring<K>::T>(
                         (std::to_integer<unsigned>(in[bit / 8]) >> shift) & ((1u << take) - 1))
                     << used;
                bit += take;
                used += take;
            }
        }
    }
}
