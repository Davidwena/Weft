#include "bj/ref/investigate.hpp"
#include <gtest/gtest.h>
#include <random>
using namespace bj;

namespace {
    uint128_t brute(const std::vector<ref::PlainRow> &in,
                    const std::vector<ref::PlainRow> &out,
                    const InvestigationConfig &c) {
        uint128_t count = 0;
        if (c.pattern == "P3") {
            for (std::size_t y = 0; y < in.size(); ++y) {
                const auto &b = in[y];
                if (b.amount < c.amount_lower || b.amount >= c.amount_upper) {
                    continue;
                }
                std::size_t d = 0;
                for (std::size_t x = 0; x < in.size(); ++x) {
                    const auto &a = in[x];
                    d += x != y && a.amount >= c.amount_lower && a.amount < c.amount_upper &&
                         a.time <= b.time && b.time - a.time <= c.windows[0].upper;
                }
                count += d >= c.min_cluster - 1;
            }
            return count;
        }
        const auto &xs = c.pattern == "P4" ? out : in;
        const auto &ys = c.pattern == "P4" ? in : out;
        for (const auto &y : ys) {
            unsigned degree = 0;
            for (const auto &x : xs) {
                degree += ref::band(static_cast<std::int64_t>(x.time),
                                    x.amount,
                                    static_cast<std::int64_t>(y.time),
                                    y.amount,
                                    c.windows[0],
                                    c.p,
                                    c.q) &&
                          (c.pattern != "P4" || x.counterparty == y.counterparty);
            }
            count += c.pattern == "P2" ? (degree ? y.amount : uint128_t{0}) : uint128_t(degree);
        }
        return count;
    }
}

TEST(T8, EfficientReferenceMatchesQuadraticAcrossAllPatterns) {
    std::mt19937_64 rng(733);
    for (unsigned trial = 0; trial < 150; ++trial) {
        std::vector<ref::PlainRow> in, out;
        for (unsigned i = 0; i < trial % 23; ++i) {
            ref::PlainRow row{1, rng() % 9, 1 + rng() % 16, false, rng() % 4};
            (i % 2 ? out : in).push_back(row);
        }
        InvestigationConfig c;
        c.q = 1 + rng() % 100;
        c.p = rng() % c.q;
        c.windows = {{0, rng() % 9}};
        c.amount_lower = 3;
        c.amount_upper = 12;
        c.min_cluster = 2 + rng() % 4;
        for (auto pattern : {"P1", "P2", "P3", "P4"}) {
            c.pattern = pattern;
            EXPECT_TRUE(ref::aggregate(in, out, c) == brute(in, out, c))
                << pattern << " trial " << trial;
        }
    }
    std::vector<ref::PlainRow> large(12, {1, 0, (uint128_t{1} << 61) + 7, false, UINT64_MAX});
    InvestigationConfig c;
    c.pattern = "P2";
    c.q = 1u << 29;
    c.p = c.q - 1;
    c.windows = {{0, 0}};
    EXPECT_TRUE(ref::aggregate(large, large, c) == brute(large, large, c));
    EXPECT_GT(ref::aggregate(large, large, c), uint128_t{UINT64_MAX});
}
