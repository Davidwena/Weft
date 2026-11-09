#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace bj {
    __extension__ using uint128_t = unsigned __int128;

    template <class T> constexpr T native_mask(unsigned k) noexcept {
        return ~T{0} >> (sizeof(T) * 8 - k);
    }

    inline uint128_t load_wide(std::span<const std::byte> b, std::size_t bit, unsigned k) noexcept;
    inline void
    store_wide(std::span<std::byte> b, std::size_t bit, unsigned k, uint128_t x) noexcept;

    inline std::size_t bits_bytes(std::size_t bits) noexcept {
        return bits / 8 + (bits % 8 != 0);
    }

    inline std::size_t bits_words(std::size_t bits) noexcept {
        return bits / 64 + (bits % 64 != 0);
    }

    inline std::uint64_t width_mask(unsigned k) noexcept {
        return UINT64_MAX >> (64 - k);
    }

    inline std::size_t checked_bits(std::size_t count, unsigned k) {
        if (!k || k > 96 || count > SIZE_MAX / k) {
            throw std::length_error("packed size overflow");
        }
        return count * k;
    }

    inline std::uint64_t
    load_packed(std::span<const std::byte> bytes, std::size_t bit, unsigned k) noexcept {
        const auto start = bit / 8;
        const unsigned shift = static_cast<unsigned>(bit % 8);
        const auto n = std::min<std::size_t>(8, bytes.size() - start);
        std::uint64_t value = 0;
        std::memcpy(&value, bytes.data() + start, n);
        value >>= shift;
        if (shift && k > 64 - shift) {
            value |= std::uint64_t(std::to_integer<unsigned>(bytes[start + 8])) << (64 - shift);
        }
        return value & width_mask(k);
    }

    inline void store_packed(std::span<std::byte> bytes,
                             std::size_t bit,
                             unsigned k,
                             std::uint64_t value) noexcept {
        const auto start = bit / 8;
        const unsigned shift = static_cast<unsigned>(bit % 8);
        const auto n = std::min<std::size_t>(8, bytes.size() - start);
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + start, n);
        const auto mask = width_mask(k);
        word = (word & ~(mask << shift)) | ((value & mask) << shift);
        std::memcpy(bytes.data() + start, &word, n);
        if (shift && k > 64 - shift) {
            const unsigned high = k - (64 - shift), high_mask = (1u << high) - 1;
            auto &last = bytes[start + 8];
            last = std::byte((std::to_integer<unsigned>(last) & ~high_mask) |
                             ((value >> (64 - shift)) & high_mask));
        }
    }

    inline void load_bits(std::span<const std::byte> bytes,
                          std::size_t offset,
                          std::size_t count,
                          std::span<std::uint64_t> out) {
        if (out.size() != bits_words(count) || offset > bytes.size() * 8 ||
            count > bytes.size() * 8 - offset) {
            throw std::invalid_argument("bit range");
        }
        for (std::size_t w = 0; w < out.size(); ++w) {
            out[w] = load_packed(bytes,
                                 offset + w * 64,
                                 static_cast<unsigned>(std::min<std::size_t>(64, count - w * 64)));
        }
    }

    inline void store_bits(std::span<std::byte> bytes,
                           std::size_t offset,
                           std::size_t count,
                           std::span<const std::uint64_t> in) {
        if (in.size() != bits_words(count) || offset > bytes.size() * 8 ||
            count > bytes.size() * 8 - offset) {
            throw std::invalid_argument("bit range");
        }
        for (std::size_t w = 0; w < in.size(); ++w) {
            store_packed(bytes,
                         offset + w * 64,
                         static_cast<unsigned>(std::min<std::size_t>(64, count - w * 64)),
                         in[w]);
        }
    }

    inline uint128_t load_wide(std::span<const std::byte> b, std::size_t bit, unsigned k) noexcept {
        const auto low = std::min(k, 64u);
        uint128_t x = load_packed(b, bit, low);
        if (k > 64) {
            x |= uint128_t(load_packed(b, bit + 64, k - 64)) << 64;
        }
        return x;
    }

    inline void
    store_wide(std::span<std::byte> b, std::size_t bit, unsigned k, uint128_t x) noexcept {
        store_packed(b, bit, std::min(k, 64u), static_cast<std::uint64_t>(x));
        if (k > 64) {
            store_packed(b, bit + 64, k - 64, static_cast<std::uint64_t>(x >> 64));
        }
    }

    template <class T>
    void unpack_native(unsigned k, std::span<const std::byte> in, std::span<T> out) {
        static_assert(std::is_unsigned_v<T> || std::is_same_v<T, uint128_t>);
        if (k > sizeof(T) * 8 || in.size() != bits_bytes(checked_bits(out.size(), k))) {
            throw std::invalid_argument("native unpack size");
        }
        if (k == sizeof(T) * 8) {
            if (!out.empty()) {
                std::memcpy(out.data(), in.data(), in.size());
            }
            return;
        }
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<T>(load_wide(in, i * k, k));
        }
    }

    template <class T>
    void pack_native(unsigned k, std::span<const T> in, std::span<std::byte> out) {
        static_assert(std::is_unsigned_v<T> || std::is_same_v<T, uint128_t>);
        if (k > sizeof(T) * 8 || out.size() != bits_bytes(checked_bits(in.size(), k))) {
            throw std::invalid_argument("native pack size");
        }
        if (k == sizeof(T) * 8) {
            if (!in.empty()) {
                std::memcpy(out.data(), in.data(), out.size());
            }
            return;
        }
        std::fill(out.begin(), out.end(), std::byte{0});
        for (std::size_t i = 0; i < in.size(); ++i) {
            store_wide(out, i * k, k, in[i]);
        }
    }
}
