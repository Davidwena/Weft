#pragma once
#include "bj/common/packed.hpp"
#include <cstdint>
#include <span>
#include <stdexcept>

namespace bj::ref {
    inline std::uint64_t b2a(bool value) noexcept {
        return value ? 1 : 0;
    }

    inline std::uint64_t and2a(std::span<const std::uint8_t> values) {
        if (values.empty() || values.size() > 4) {
            throw std::invalid_argument("reference AND arity");
        }
        std::uint64_t product = 1;
        for (auto bit : values) {
            if (bit > 1) {
                throw std::invalid_argument("reference AND requires bits");
            }
            product &= bit;
        }
        return product;
    }

    template <class T>
    void ltbits(std::span<const T> c,
                std::span<const T> r,
                std::span<std::uint8_t> out,
                bool reverse = false) {
        if (c.size() != r.size() || c.size() != out.size()) {
            throw std::invalid_argument("reference LTBits shape");
        }
        for (std::size_t i = 0; i < c.size(); ++i) {
            out[i] = reverse ? r[i] < c[i] : c[i] < r[i];
        }
    }

    template <class T> void msb(unsigned k, std::span<const T> x, std::span<std::uint8_t> out) {
        if (!k || k > sizeof(T) * 8 || x.size() != out.size()) {
            throw std::invalid_argument("reference MSB shape");
        }
        for (std::size_t i = 0; i < x.size(); ++i) {
            out[i] = static_cast<std::uint8_t>((x[i] >> (k - 1)) & 1);
        }
    }

    template <class T> void eq(std::span<const T> x, std::span<std::uint8_t> out) {
        if (x.size() != out.size()) {
            throw std::invalid_argument("reference EQ shape");
        }
        for (std::size_t i = 0; i < x.size(); ++i) {
            out[i] = x[i] == 0;
        }
    }

    template <class T>
    void lts(std::span<const T> x, std::span<const T> y, std::span<std::uint8_t> out) {
        ltbits<T>(x, y, out);
    }

}
