#pragma once
#include "bj/ref/band.hpp"
#include <algorithm>
#include <numeric>

namespace bj::ref {
    struct DCSlot {
        std::int64_t time = 0;
        uint128_t amount = 0;
        bool valid = true;
    };

    struct DCElement {
        __int128_t time, a1, a2;
        std::size_t id, rank = 0;
        bool pt, ps, ng;
    };

    inline std::vector<DCElement> dc_elements(std::span<const DCSlot> in,
                                              std::span<const DCSlot> out,
                                              const BandWindow &w,
                                              std::uint64_t p,
                                              std::uint64_t q) {
        if (w.lower > w.upper || p >= q) {
            throw std::invalid_argument("reference DC invalid parameters");
        }
        std::vector<DCElement> e;
        for (auto x : in) {
            e.push_back({3 * __int128_t(x.time) + 1,
                         3 * __int128_t(q - p) * __int128_t(x.amount) + 1,
                         3 * __int128_t(q + p) * __int128_t(x.amount) + 1,
                         e.size(),
                         0,
                         x.valid,
                         false,
                         false});
        }
        for (unsigned sign = 0; sign < 2; ++sign) {
            for (auto x : out) {
                e.push_back(
                    {3 * (__int128_t(x.time) - __int128_t(sign ? w.upper + 1 : w.lower)) + 2,
                     3 * __int128_t(q) * __int128_t(x.amount) + 2,
                     3 * __int128_t(q) * __int128_t(x.amount),
                     e.size(),
                     0,
                     false,
                     sign == 0 && x.valid,
                     sign == 1 && x.valid});
            }
        }
        return e;
    }

    inline std::int64_t dominance_brute(const std::vector<DCElement> &e, unsigned r) {
        std::int64_t count = 0;
        for (auto x : e) {
            if (x.pt) {
                for (auto y : e) {
                    if (x.time < y.time && (r ? x.a2 < y.a2 : x.a1 < y.a1)) {
                        count += std::int64_t(y.ps) - std::int64_t(y.ng);
                    }
                }
            }
        }
        return count;
    }

    inline std::int64_t dominance_partition(std::vector<DCElement> e, unsigned r) {
        std::sort(e.begin(), e.end(), [](auto x, auto y) {
            return std::pair{x.time, x.id} < std::pair{y.time, y.id};
        });
        for (std::size_t i = 0; i < e.size(); ++i) {
            e[i].rank = i;
        }
        std::sort(e.begin(), e.end(), [r](auto x, auto y) {
            return std::pair{r ? x.a2 : x.a1, x.id} < std::pair{r ? y.a2 : y.a1, y.id};
        });
        std::int64_t acc = 0;
        const auto h = std::bit_width(e.empty() ? 0 : e.size() - 1);
        for (unsigned it = 0; it < h; ++it) {
            const auto ell = h - 1 - it, half = std::size_t{1} << ell, block = 2 * half;
            for (std::size_t start = 0; start < e.size(); start += block) {
                const auto end = std::min(e.size(), start + block);
                std::int64_t points = 0;
                for (std::size_t j = start; j < end; ++j) {
                    const bool bit = (e[j].rank >> ell) & 1;
                    if (bit) {
                        acc += (std::int64_t(e[j].ps) - std::int64_t(e[j].ng)) * points;
                    } else {
                        points += e[j].pt;
                    }
                }
                if (ell) {
                    std::stable_partition(e.begin() + start, e.begin() + end, [ell](auto x) {
                        return !((x.rank >> ell) & 1);
                    });
                }
            }
        }
        return acc;
    }

    inline std::uint64_t dc_brute(std::span<const DCSlot> in,
                                  std::span<const DCSlot> out,
                                  const BandWindow &w,
                                  std::uint64_t p,
                                  std::uint64_t q) {
        std::uint64_t count = 0;
        for (auto x : in) {
            for (auto y : out) {
                count += x.valid && y.valid && band(x.time, x.amount, y.time, y.amount, w, p, q);
            }
        }
        return count;
    }
}
