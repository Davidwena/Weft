#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <wmmintrin.h>

namespace bj {

    class Prg {
        alignas(16) __m128i keys_[11];

    public:
        using Key = std::array<std::byte, 16>;
        explicit Prg(const Key &key) noexcept;
        void encrypt_block(std::span<const std::byte, 16> in,
                           std::span<std::byte, 16> out) const noexcept;
        void fill(std::uint64_t domain, std::uint64_t byte_offset, std::span<std::byte> out) const;
    };
}
