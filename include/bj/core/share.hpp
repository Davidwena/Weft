#pragma once
#include "bj/common/ring.hpp"
#include "bj/common/packed.hpp"

namespace bj {
    template <unsigned K> struct MaskedCol {
        using T = typename Ring<K>::T;
        std::span<const T> Delta, lambda_share;
    };

    template <unsigned K, class Op>
    void ring_binary(std::span<const typename Ring<K>::T> a,
                     std::span<const typename Ring<K>::T> b,
                     std::span<typename Ring<K>::T> out,
                     Op op) {
        if (a.size() != b.size() || a.size() != out.size()) {
            throw std::invalid_argument("share column size");
        }
#pragma omp parallel for if (a.size() >= 16384) schedule(static)
        for (std::size_t i = 0; i < a.size(); ++i) {
            out[i] = op(a[i], b[i]);
        }
    }

    template <unsigned K>
    void share_add(std::span<const typename Ring<K>::T> a,
                   std::span<const typename Ring<K>::T> b,
                   std::span<typename Ring<K>::T> out) {
        ring_binary<K>(a, b, out, [](auto x, auto y) {
            return Ring<K>::add(x, y);
        });
    }

    template <unsigned K>
    void share_sub(std::span<const typename Ring<K>::T> a,
                   std::span<const typename Ring<K>::T> b,
                   std::span<typename Ring<K>::T> out) {
        ring_binary<K>(a, b, out, [](auto x, auto y) {
            return Ring<K>::sub(x, y);
        });
    }

    template <unsigned K>
    void share_scale(std::span<const typename Ring<K>::T> in,
                     typename Ring<K>::T c,
                     std::span<typename Ring<K>::T> out) {
        if (in.size() != out.size()) {
            throw std::invalid_argument("share column size");
        }
#pragma omp parallel for if (in.size() >= 16384) schedule(static)
        for (std::size_t i = 0; i < in.size(); ++i) {
            out[i] = Ring<K>::mul(in[i], c);
        }
    }

    template <unsigned K>
    void share_add_public(std::span<const typename Ring<K>::T> in,
                          typename Ring<K>::T c,
                          unsigned party,
                          std::span<typename Ring<K>::T> out) {
        if (in.size() != out.size()) {
            throw std::invalid_argument("share column size");
        }
        const auto value = party == 0 ? c : 0;
#pragma omp parallel for if (in.size() >= 16384) schedule(static)
        for (std::size_t i = 0; i < in.size(); ++i) {
            out[i] = Ring<K>::add(in[i], value);
        }
    }

    template <unsigned K, unsigned K2>
    void share_reduce(std::span<const typename Ring<K>::T> in,
                      std::span<typename Ring<K2>::T> out) {
        static_assert(K2 <= K);
        if (in.size() != out.size()) {
            throw std::invalid_argument("share column size");
        }
#pragma omp parallel for if (in.size() >= 16384) schedule(static)
        for (std::size_t i = 0; i < in.size(); ++i) {
            out[i] = Ring<K>::template reduce<K2>(in[i]);
        }
    }

    template <unsigned K>
    void masked_to_additive(MaskedCol<K> col, unsigned party, std::span<typename Ring<K>::T> out) {
        if (col.Delta.size() != out.size() || col.lambda_share.size() != out.size()) {
            throw std::invalid_argument("masked column size");
        }
#pragma omp parallel for if (out.size() >= 16384) schedule(static)
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = Ring<K>::sub(party == 0 ? col.Delta[i] : 0, col.lambda_share[i]);
        }
    }

    inline void xor_shares(std::span<const std::uint64_t> a,
                           std::span<const std::uint64_t> b,
                           std::span<std::uint64_t> out) {
        if (a.size() != b.size() || a.size() != out.size()) {
            throw std::invalid_argument("xor column size");
        }
#pragma omp parallel for if (out.size() >= 16384) schedule(static)
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = a[i] ^ b[i];
        }
    }

    inline void xor_public(std::span<const std::uint64_t> in,
                           std::span<const std::uint64_t> value,
                           unsigned party,
                           std::span<std::uint64_t> out) {
        if (in.size() != value.size() || in.size() != out.size()) {
            throw std::invalid_argument("xor public size");
        }
#pragma omp parallel for if (out.size() >= 16384) schedule(static)
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = in[i] ^ (party == 0 ? value[i] : 0);
        }
    }
}
