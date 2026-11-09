#include "bj/common/prg.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace bj {
    namespace {
        template <int R> __m128i next_key(__m128i k) noexcept {
            auto a = _mm_aeskeygenassist_si128(k, R);
            a = _mm_shuffle_epi32(a, 0xff);
            k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
            k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
            k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
            return _mm_xor_si128(k, a);
        }

        __m128i counter(std::uint64_t domain, std::uint64_t index) noexcept {
            return _mm_set_epi64x(static_cast<long long>(__builtin_bswap64(index)),
                                  static_cast<long long>(__builtin_bswap64(domain)));
        }
    }

    Prg::Prg(const Key &key) noexcept {
        keys_[0] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(key.data()));
        keys_[1] = next_key<0x01>(keys_[0]);
        keys_[2] = next_key<0x02>(keys_[1]);
        keys_[3] = next_key<0x04>(keys_[2]);
        keys_[4] = next_key<0x08>(keys_[3]);
        keys_[5] = next_key<0x10>(keys_[4]);
        keys_[6] = next_key<0x20>(keys_[5]);
        keys_[7] = next_key<0x40>(keys_[6]);
        keys_[8] = next_key<0x80>(keys_[7]);
        keys_[9] = next_key<0x1b>(keys_[8]);
        keys_[10] = next_key<0x36>(keys_[9]);
    }

    void Prg::encrypt_block(std::span<const std::byte, 16> in,
                            std::span<std::byte, 16> out) const noexcept {
        auto v =
            _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(in.data())), keys_[0]);
        for (int r = 1; r < 10; ++r) {
            v = _mm_aesenc_si128(v, keys_[r]);
        }
        v = _mm_aesenclast_si128(v, keys_[10]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out.data()), v);
    }

    void Prg::fill(std::uint64_t domain, std::uint64_t offset, std::span<std::byte> out) const {
        if (out.size() > std::numeric_limits<std::uint64_t>::max() - offset) {
            throw std::length_error("CTR offset overflow");
        }
        auto block = offset / 16;
        const auto skip = offset % 16;
        if (skip && !out.empty()) {
            alignas(16) std::array<std::byte, 16> tmp{};
            auto c = counter(domain, block++);
            encrypt_block(
                std::span<const std::byte, 16>(reinterpret_cast<const std::byte *>(&c), 16), tmp);
            const auto n = std::min<std::size_t>(16 - skip, out.size());
            std::memcpy(out.data(), tmp.data() + skip, n);
            out = out.subspan(n);
        }
        while (out.size() >= 128) {
            __m128i x[8];
            for (unsigned i = 0; i < 8; ++i) {
                x[i] = _mm_xor_si128(counter(domain, block + i), keys_[0]);
            }
            for (unsigned r = 1; r < 10; ++r) {
                for (auto &v : x) {
                    v = _mm_aesenc_si128(v, keys_[r]);
                }
            }
            for (unsigned i = 0; i < 8; ++i) {
                _mm_storeu_si128(reinterpret_cast<__m128i *>(out.data() + 16 * i),
                                 _mm_aesenclast_si128(x[i], keys_[10]));
            }
            block += 8;
            out = out.subspan(128);
        }
        while (!out.empty()) {
            alignas(16) std::array<std::byte, 16> tmp{};
            auto c = counter(domain, block++);
            encrypt_block(
                std::span<const std::byte, 16>(reinterpret_cast<const std::byte *>(&c), 16), tmp);
            const auto n = std::min<std::size_t>(16, out.size());
            std::memcpy(out.data(), tmp.data(), n);
            out = out.subspan(n);
        }
    }
}
