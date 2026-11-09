#pragma once
#include "bj/plan/investigation.hpp"

namespace bj::ref {
    struct PlainRow {
        std::uint64_t subject = 0, time = 0;
        uint128_t amount = 0;
        bool outgoing = false;
        std::uint64_t counterparty = 0;
    };

    inline bool band(std::int64_t tin,
                     uint128_t ain,
                     std::int64_t tout,
                     uint128_t aout,
                     const BandWindow &w,
                     std::uint64_t p,
                     std::uint64_t q) {
        const auto g = static_cast<__int128_t>(tout) - tin;
        return g >= w.lower && g <= w.upper && q * aout <= (q + p) * ain &&
               (q - p) * ain <= q * aout;
    }

    struct IdealResult {
        bool overflow = false;
        std::vector<std::uint64_t> counts;
        std::vector<PlainRow> inbound, outbound;
    };

    inline IdealResult investigation(std::span<const PlainRow> rows,
                                     std::uint64_t subject,
                                     const InvestigationConfig &c) {
        IdealResult r;
        std::size_t ni = 0, no = 0;
        for (const auto &row : rows) {
            if (row.subject == subject) {
                if (row.outgoing) {
                    ++no;
                    if (r.outbound.size() < c.kout) {
                        r.outbound.push_back(row);
                    }
                } else {
                    ++ni;
                    if (r.inbound.size() < c.kin) {
                        r.inbound.push_back(row);
                    }
                }
            }
        }
        r.overflow = ni > c.kin || (c.pattern != "P3" && no > c.kout);
        r.counts.resize(c.windows.size());
        for (std::size_t w = 0; w < c.windows.size(); ++w) {
            if (c.pattern == "P3") {
                for (std::size_t j = 0; j < r.inbound.size(); ++j) {
                    auto y = r.inbound[j];
                    if (y.amount < c.amount_lower || y.amount >= c.amount_upper) {
                        continue;
                    }
                    std::size_t degree = 0;
                    for (std::size_t i = 0; i < r.inbound.size(); ++i) {
                        auto x = r.inbound[i];
                        degree += i != j && x.amount >= c.amount_lower &&
                                  x.amount < c.amount_upper && x.time <= y.time &&
                                  y.time - x.time <= c.windows[w].upper;
                    }
                    r.counts[w] += degree >= c.min_cluster - 1;
                }
            } else {
                const auto &xs = c.pattern == "P4" ? r.outbound : r.inbound;
                const auto &ys = c.pattern == "P4" ? r.inbound : r.outbound;
                for (auto y : ys) {
                    std::uint64_t degree = 0;
                    for (auto x : xs) {
                        degree += band(static_cast<std::int64_t>(x.time),
                                       x.amount,
                                       static_cast<std::int64_t>(y.time),
                                       y.amount,
                                       c.windows[w],
                                       c.p,
                                       c.q) &&
                                  (c.pattern != "P4" || x.counterparty == y.counterparty);
                    }
                    r.counts[w] += c.pattern == "P2"
                                       ? (degree ? static_cast<std::uint64_t>(y.amount) : 0ull)
                                       : degree;
                }
            }
        }
        return r;
    }
}
