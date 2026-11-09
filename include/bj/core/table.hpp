#pragma once
#include "bj/core/input.hpp"
#include "bj/plan/investigation.hpp"

namespace bj {

    struct MaskedColumn {
        unsigned width;
        std::size_t count;
        bool dealer_knows_mask = true;
        AlignedBuffer<std::uint32_t> d32, m32;
        AlignedBuffer<std::uint64_t> d64, m64;
        AlignedBuffer<uint128_t> d128, m128;
        BitVec bits, mask_bits;

        MaskedColumn(unsigned k, std::size_t n)
            : width(k), count(n), d32(k && k <= 32 ? n : 0), m32(d32.size()),
              d64(k > 32 && k <= 64 ? n : 0), m64(d64.size()), d128(k > 64 ? n : 0),
              m128(d128.size()), bits(k ? 0 : n), mask_bits(k ? 0 : n) {
            zero();
        }

        template <class T> std::span<T> delta() {
            if constexpr (sizeof(T) == 4) {
                return d32.span();
            } else if constexpr (sizeof(T) == 8) {
                return d64.span();
            } else {
                return d128.span();
            }
        }

        template <class T> std::span<const T> delta() const {
            if constexpr (sizeof(T) == 4) {
                return d32.span();
            } else if constexpr (sizeof(T) == 8) {
                return d64.span();
            } else {
                return d128.span();
            }
        }

        template <class T> std::span<T> mask() {
            if constexpr (sizeof(T) == 4) {
                return m32.span();
            } else if constexpr (sizeof(T) == 8) {
                return m64.span();
            } else {
                return m128.span();
            }
        }

        template <class T> std::span<const T> mask() const {
            if constexpr (sizeof(T) == 4) {
                return m32.span();
            } else if constexpr (sizeof(T) == 8) {
                return m64.span();
            } else {
                return m128.span();
            }
        }

        void zero() {
            std::fill(d32.span().begin(), d32.span().end(), 0);
            std::fill(m32.span().begin(), m32.span().end(), 0);
            std::fill(d64.span().begin(), d64.span().end(), 0);
            std::fill(m64.span().begin(), m64.span().end(), 0);
            std::fill(d128.span().begin(), d128.span().end(), 0);
            std::fill(m128.span().begin(), m128.span().end(), 0);
            std::fill(bits.words().begin(), bits.words().end(), 0);
            std::fill(mask_bits.words().begin(), mask_bits.words().end(), 0);
        }
    };

    struct MaskedTable {
        std::vector<MaskedColumn> columns;
        bool dealer_knows_positions = true;

        MaskedTable(std::size_t n, std::span<const unsigned> widths) {
            columns.reserve(widths.size());
            for (auto k : widths) {
                columns.emplace_back(k, n);
            }
        }

        void mark_public_reordered() noexcept {
            dealer_knows_positions = false;
            for (auto &c : columns) {
                c.dealer_knows_mask = false;
            }
        }
    };

    inline void load_mask_share(MaskedColumn &c,
                                MaterialStore &s,
                                const MaterialPlan &p,
                                std::size_t combined) {
        if (combined < p.inputs.size()) {
            throw std::invalid_argument("expected fresh positional mask");
        }
        auto segment = s.layout().masks.at(combined - p.inputs.size());
        if (segment == SIZE_MAX) {
            throw std::invalid_argument("mask is local/undistributed");
        }
        const auto &spec = p.masks.at(combined - p.inputs.size());
        if (spec.width != c.width || spec.count != c.count) {
            throw std::invalid_argument("mask column shape");
        }
        s.claim_mask(combined - p.inputs.size());
        const auto bytes = s.packed(segment);
        if (!c.width) {
            c.mask_bits.unpack(bytes);
        } else if (c.width <= 32) {
            unpack_native<std::uint32_t>(c.width, bytes, c.m32.span());
        } else if (c.width <= 64) {
            unpack_native<std::uint64_t>(c.width, bytes, c.m64.span());
        } else {
            unpack_native<uint128_t>(c.width, bytes, c.m128.span());
        }
    }
}
