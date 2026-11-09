#pragma once

#include "bj/core/comparison_dealer.hpp"
#include "bj/common/prg.hpp"
#include <omp.h>
#include <cstring>
#include <atomic>

namespace bj {
    inline bool x11_parallel(std::size_t n) {
        return n >= 131072 && !omp_in_parallel();
    }

    template <class T> void x11_pack(unsigned k, std::span<const T> in, std::span<std::byte> out) {
        if (!x11_parallel(in.size())) {
            pack_native<T>(k, in, out);
            return;
        }
        if (out.size() != bits_bytes(checked_bits(in.size(), k))) {
            throw std::invalid_argument("parallel pack size");
        }
        constexpr std::size_t tile = 65536;
#pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < (in.size() + tile - 1) / tile; ++j) {
            auto start = j * tile, n = std::min(tile, in.size() - start);
            pack_native<T>(k, in.subspan(start, n), out.subspan(start * k / 8, bits_bytes(n * k)));
        }
    }

    template <class T> void x11_pack(unsigned k, std::span<T> in, std::span<std::byte> out) {
        x11_pack<T>(k, std::span<const T>(in), out);
    }

    template <class T>
    void x11_unpack(unsigned k, std::span<const std::byte> in, std::span<T> out) {
        if (!x11_parallel(out.size())) {
            unpack_native<T>(k, in, out);
            return;
        }
        if (in.size() != bits_bytes(checked_bits(out.size(), k))) {
            throw std::invalid_argument("parallel unpack size");
        }
        constexpr std::size_t tile = 65536;
#pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < (out.size() + tile - 1) / tile; ++j) {
            auto start = j * tile, n = std::min(tile, out.size() - start);
            unpack_native<T>(
                k, in.subspan(start * k / 8, bits_bytes(n * k)), out.subspan(start, n));
        }
    }

    inline void x11_prg(const Prg &prg, std::uint64_t domain, std::span<std::byte> out) {
        constexpr std::size_t tile = 1 << 20;
#pragma omp parallel for if (x11_parallel(out.size())) schedule(static)
        for (std::size_t j = 0; j < (out.size() + tile - 1) / tile; ++j) {
            auto start = j * tile;
            prg.fill(domain, start, out.subspan(start, std::min(tile, out.size() - start)));
        }
    }

    template <class T>
    void
    x11_table(const CompareShape &shape, std::span<const T> thresholds, std::span<std::byte> out) {
        const auto n = thresholds.size(), entries = std::size_t{1} << shape.d;
        if (entries < 8 || !x11_parallel(n)) {
            make_comparison_table<T>(shape, thresholds, out);
            return;
        }
        if (out.size() != bits_bytes(shape.table_bits() * n)) {
            throw std::invalid_argument("parallel table size");
        }
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = std::byte{0};
        }

#pragma omp parallel for collapse(2) schedule(static)
        for (unsigned b = 0; b < shape.blocks; ++b) {
            for (std::size_t i = 0; i < n; ++i) {
                auto digit = (thresholds[i] >> ((shape.blocks - 1 - b) * shape.d)) & (entries - 1);
                auto bit = (b * n + i) * entries + digit;
                out[bit / 8] |= std::byte(1u << (bit % 8));
            }
        }
    }

    inline void x11_tile_evaluation(const CompareShape &s,
                                    std::size_t n,
                                    std::span<std::byte> out,
                                    const ComparisonRandom &random) {
        std::fill(out.begin(), out.end(), std::byte{0});
        auto masks = out.first(bits_bytes(s.inputs * n));
        random(masks);
        if (s.inputs * n % 8) {
            masks.back() &= std::byte((1u << (s.inputs * n % 8)) - 1);
        }
        std::vector<std::pair<std::size_t, unsigned>> dependencies;
        for (auto subset : s.products) {
            auto bit = static_cast<unsigned>(std::countr_zero(subset));
            auto rest = subset & (subset - 1);
            std::size_t left;
            if (std::has_single_bit(rest)) {
                left = std::countr_zero(rest);
            } else {
                auto it = std::lower_bound(s.products.begin(), s.products.end(), rest);
                if (it == s.products.end() || *it != rest) {
                    throw std::logic_error("missing subset recurrence");
                }
                left = s.inputs + static_cast<std::size_t>(it - s.products.begin());
            }
            dependencies.emplace_back(left, bit);
        }
        constexpr std::size_t words = 16;
        AlignedBuffer<std::uint64_t> values((s.inputs + s.products.size()) * words);
        for (std::size_t begin = 0; begin < n; begin += 64 * words) {
            auto nw = bits_words(std::min(n - begin, 64 * words));
            for (unsigned j = 0; j < s.inputs; ++j) {
                for (std::size_t w = 0; w < nw; ++w) {
                    auto at = begin + 64 * w;
                    values.data()[j * words + w] =
                        load_packed(masks,
                                    j * n + at,
                                    static_cast<unsigned>(std::min<std::size_t>(64, n - at)));
                }
            }
            for (std::size_t p = 0; p < s.products.size(); ++p) {
                auto [left, right] = dependencies[p];
                for (std::size_t w = 0; w < nw; ++w) {
                    auto value = values.data()[left * words + w] & values.data()[right * words + w];
                    values.data()[(s.inputs + p) * words + w] = value;
                    auto at = begin + 64 * w;
                    store_packed(out,
                                 (s.inputs + p) * n + at,
                                 static_cast<unsigned>(std::min<std::size_t>(64, n - at)),
                                 value);
                }
            }
        }
    }

    inline void x11_evaluation(const CompareShape &s,
                               std::size_t n,
                               std::span<std::byte> out,
                               const ComparisonRandom &random) {
        if (!x11_parallel(n)) {
            make_comparison_evaluation(s, n, out, random);
            return;
        }
        if (out.size() != bits_bytes(s.eval_bits() * n)) {
            throw std::invalid_argument("parallel evaluation size");
        }
        std::fill(out.begin(), out.end(), std::byte{0});
        auto masks = out.first(bits_bytes(s.inputs * n));
        random(masks);
        if (s.inputs * n % 8) {
            masks.back() &= std::byte((1u << (s.inputs * n % 8)) - 1);
        }

        AlignedBuffer<std::byte> rho(masks.size());
        std::copy(masks.begin(), masks.end(), rho.span().begin());
        auto scatter = [&](std::span<const std::byte> src,
                           std::size_t from,
                           std::size_t to,
                           std::size_t count) {
            auto edge = [&](unsigned take) {
                auto v = std::byte(load_packed(src, from, take) << (to % 8));
                std::atomic_ref<std::byte> dst(out[to / 8]);
                auto old = dst.load(std::memory_order_relaxed);
                while (!dst.compare_exchange_weak(old, old | v, std::memory_order_relaxed)) {
                }
                from += take;
                to += take;
                count -= take;
            };
            if (to % 8) {
                edge(static_cast<unsigned>(std::min<std::size_t>(count, 8 - to % 8)));
            }
            while (count >= 64) {
                auto word = load_packed(src, from, 64);
                std::memcpy(out.data() + to / 8, &word, 8);
                from += 64;
                to += 64;
                count -= 64;
            }
            while (count >= 8) {
                out[to / 8] = std::byte(load_packed(src, from, 8));
                from += 8;
                to += 8;
                count -= 8;
            }
            if (count) {
                edge(static_cast<unsigned>(count));
            }
        };
        constexpr std::size_t tile = 65536;
#pragma omp parallel
        {
            AlignedBuffer<std::byte> scratch(bits_bytes(s.eval_bits() * tile));
#pragma omp for schedule(static)
            for (std::size_t t = 0; t < (n + tile - 1) / tile; ++t) {
                auto begin = t * tile, count = std::min(tile, n - begin);
                auto local = scratch.span().first(bits_bytes(s.eval_bits() * count));
                x11_tile_evaluation(s, count, local, [&](auto target) {
                    std::fill(target.begin(), target.end(), std::byte{0});
                    for (unsigned j = 0; j < s.inputs; ++j) {
                        for (std::size_t k = 0; k < count; k += 64) {
                            auto take = static_cast<unsigned>(std::min<std::size_t>(64, count - k));
                            store_packed(target,
                                         j * count + k,
                                         take,
                                         load_packed(rho.span(), j * n + begin + k, take));
                        }
                    }
                });
                for (std::size_t p = 0; p < s.products.size(); ++p) {
                    scatter(local, (s.inputs + p) * count, (s.inputs + p) * n + begin, count);
                }
            }
        }
    }

    inline void x11_mask_term(const MaskSpec &spec,
                              const MaskTerm &term,
                              std::span<const std::byte> source,
                              unsigned source_k,
                              std::span<std::byte> out) {
        const auto k = spec.width ? spec.width : 1;
        auto tile = [&](std::size_t start,
                        std::size_t count,
                        std::span<std::byte> target,
                        std::size_t destbase) {
            for (std::size_t j = start; j < start + count; ++j) {
                auto i = term.destination_offset + j,
                     from = term.source_offset + (j / term.divisor) % term.modulus;
                auto x = load_wide(source, from * source_k, source_k),
                     old = load_wide(target, (i - destbase) * k, k);
                auto v = spec.width ? (old + static_cast<uint128_t>(term.coefficient) * x) &
                                          native_mask<uint128_t>(k)
                                    : (old ^ x);
                store_wide(target, (i - destbase) * k, k, v);
            }
        };
        if (!x11_parallel(term.length) || (term.destination_offset * k) % 8) {
            tile(0, term.length, out, 0);
            return;
        }
        constexpr std::size_t count = 65536;
#pragma omp parallel for schedule(static)
        for (std::size_t t = 0; t < (term.length + count - 1) / count; ++t) {
            auto start = t * count, n = std::min(count, term.length - start),
                 base = term.destination_offset + start;
            tile(start, n, out.subspan(base * k / 8, bits_bytes(n * k)), base);
        }
    }

    inline void x11_correction(unsigned width,
                               std::size_t n,
                               std::span<const std::uint32_t> perm,
                               std::span<const std::byte> next,
                               std::span<const std::byte> previous,
                               std::span<std::byte> correction) {
        if (width == 32 || width == 64 || width == 96) {
#pragma omp parallel for if (x11_parallel(n)) schedule(static)
            for (std::size_t i = 0; i < n; ++i) {
                auto from = std::size_t(perm[i]);
                if (width == 32) {
                    std::uint32_t a, b;
                    std::memcpy(&a, next.data() + 4 * i, 4);
                    std::memcpy(&b, previous.data() + 4 * from, 4);
                    a -= b;
                    std::memcpy(correction.data() + 4 * i, &a, 4);
                } else {
                    auto stride = width == 64 ? 8ul : 12ul;
                    std::uint64_t a, b;
                    std::memcpy(&a, next.data() + stride * i, 8);
                    std::memcpy(&b, previous.data() + stride * from, 8);
                    auto value = a - b;
                    std::memcpy(correction.data() + stride * i, &value, 8);
                    if (width == 96) {
                        std::uint32_t hi, old;
                        std::memcpy(&hi, next.data() + stride * i + 8, 4);
                        std::memcpy(&old, previous.data() + stride * from + 8, 4);
                        hi -= old + std::uint32_t(a < b);
                        std::memcpy(correction.data() + stride * i + 8, &hi, 4);
                    }
                }
            }
            return;
        }
        auto k = width ? width : 1;
        constexpr std::size_t tile = 65536;
#pragma omp parallel for if (x11_parallel(n)) schedule(static)
        for (std::size_t t = 0; t < (n + tile - 1) / tile; ++t) {
            auto start = t * tile, count = std::min(tile, n - start);
            auto dest = correction.subspan(start * k / 8, bits_bytes(count * k));
            for (std::size_t i = start; i < start + count; ++i) {
                auto a = load_wide(next, i * k, k),
                     b = load_wide(previous, std::size_t(perm[i]) * k, k);
                store_wide(
                    dest, (i - start) * k, k, width ? (a - b) & native_mask<uint128_t>(k) : a ^ b);
            }
        }
    }
}
