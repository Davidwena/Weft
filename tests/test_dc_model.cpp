#include "bj/ops/quicksort.hpp"
#include "bj/ref/dc.hpp"
#include <gtest/gtest.h>
#include <random>

namespace {
    using namespace bj;

    TEST(DCModel, ReductionAndStablePartitionAgainstBrute) {
        std::mt19937_64 rng(88613);
        unsigned nonzero = 0;
        for (unsigned trial = 0; trial < 600; ++trial) {
            const auto ni = trial % 13, no = (trial / 13) % 11;
            std::vector<ref::DCSlot> in(ni), out(no);
            for (auto *slots : {&in, &out}) {
                for (auto &x : *slots) {
                    x.time = static_cast<std::int64_t>(rng() % 8);
                    x.amount = rng() % 5;
                    x.valid = (rng() % 5) != 0;
                }
            }
            std::uint64_t p = trial % 3 == 0 ? 0 : 3, q = 4, L = trial % 4 == 0 ? 0 : rng() % 4,
                          U = L + (trial % 5 == 0 ? 0 : rng() % 5);
            if (trial % 7 == 0) {
                L = U = 0;
            }
            const BandWindow w{L, U};
            if (trial % 23 == 0) {
                for (auto &x : in) {
                    x.valid = false;
                }
            }
            auto e = ref::dc_elements(in, out, w, p, q);
            const auto ideal = ref::dc_brute(in, out, w, p, q);
            const auto a = ref::dominance_brute(e, 0), b = ref::dominance_brute(e, 1);
            EXPECT_EQ(a - b, ideal) << trial;
            EXPECT_EQ(ref::dominance_partition(e, 0), a) << trial;
            EXPECT_EQ(ref::dominance_partition(e, 1), b) << trial;
            nonzero += ideal != 0;
        }
        EXPECT_GT(nonzero, 250u);
        EXPECT_THROW(ref::dc_elements({}, {}, {3, 2}, 0, 1), std::invalid_argument);
    }

    TEST(DCPlan, RangeChecksAndFinitePoolCertificate) {
        InvestigationConfig c;
        c.algorithm = "dc";
        c.rows = {0, 0, 0};
        c.mode = "band";
        c.kin = c.kout = 32768;
        EXPECT_NO_THROW(c.validate());
        const auto N = c.kin + 2 * c.kout;
        const auto cap = dc_pool_capacity(N, 1.5, true);
        EXPECT_EQ(cap, dc_pool_capacity(N, 1.5, false));
        EXPECT_LE(cap, N * (N - 1) / 2);
        for (std::size_t n = 0; n < 64; ++n) {
            EXPECT_GE(dc_pool_capacity(n, 1.5, true), n ? n - 1 : 0);
        }
        auto larger = c;
        larger.kin = larger.kout = 65536;
        EXPECT_NO_THROW(larger.validate());
        larger.output = "alarm";
        EXPECT_THROW(larger.validate(), std::invalid_argument);
        larger = c;
        larger.kin = larger.kout = 4;
        larger.ba = 62;
        larger.bq = 1;
        larger.p = 0;
        larger.q = 1;
        EXPECT_NO_THROW(larger.validate());
        auto bad = c;
        bad.windows = {{10, 9}};
        EXPECT_THROW(bad.validate(), std::invalid_argument);
        bad = c;
        bad.ba = 50;
        bad.ka = 59;
        EXPECT_THROW(bad.validate(), std::invalid_argument);
        bad = c;
        bad.bt = 44;
        bad.kt = 64;
        EXPECT_THROW(bad.validate(), std::invalid_argument);
        bad = c;
        bad.dc_pool_factor = 0;
        EXPECT_THROW(bad.validate(), std::invalid_argument);
    }

    TEST(DCPlan, BoundedLayoutNeverExceedsPreallocatedChunks) {
        OpenLayout l;
        l.add_ring(72, 98304);
        l.add_bits(17 * 98304);
        l.seal(true);
        const auto cap = l.chunks().size();
        for (auto n : {0ul, 1ul, 65ul, 32767ul, 98304ul}) {
            const std::array<std::size_t, 2> counts{n, 17 * n};
            l.resize_counts(counts);
            EXPECT_LE(l.chunks().size(), cap);
            EXPECT_EQ(l.bits(), 89 * n);
        }
        const std::array<std::size_t, 2> bad{98305, 0};
        EXPECT_THROW(l.resize_counts(bad), std::length_error);
    }

    TEST(DCPlan, ExactSmallQuicksortMoments) {
        for (unsigned n = 2; n <= 8; ++n) {
            std::vector<unsigned> a(n);
            std::iota(a.begin(), a.end(), 0);
            long double sum = 0, square = 0, trials = 0;
            do {
                auto b = a;
                auto quick = [&](auto &&self, std::size_t lo, std::size_t hi) -> unsigned {
                    if (hi - lo < 2) {
                        return 0;
                    }
                    const auto pivot = b[lo];
                    auto end = std::partition(b.begin() + lo + 1, b.begin() + hi, [pivot](auto x) {
                        return x < pivot;
                    });
                    const auto pos = static_cast<std::size_t>(end - b.begin()) - 1;
                    std::swap(b[lo], b[pos]);
                    return static_cast<unsigned>(hi - lo - 1) + self(self, lo, pos) +
                           self(self, pos + 1, hi);
                };
                const auto cost = quick(quick, 0, n);
                sum += cost;
                square += cost * cost;
                ++trials;
            } while (std::next_permutation(a.begin(), a.end()));
            long double H = 0, H2 = 0;
            for (unsigned i = 1; i <= n; ++i) {
                H += 1.L / i;
                H2 += 1.L / (i * i);
            }
            const auto mean = 2 * (n + 1) * H - 4 * n,
                       var = 7.L * n * n - 4.L * (n + 1) * (n + 1) * H2 - 2 * (n + 1) * H + 13 * n;
            EXPECT_NEAR(double(sum / trials), double(mean), 1e-10);
            EXPECT_NEAR(double(square / trials - mean * mean), double(var), 1e-10);
        }
    }
}
