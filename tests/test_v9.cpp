#include "bj/ops/quicksort.hpp"
#include "bj/ref/investigate.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <nlohmann/json.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
using namespace bj;

TEST(V9, ExactExpectedBudgetEveryPublicSize) {
    using Wide = boost::multiprecision::cpp_dec_float_100;

    EXPECT_EQ(dc_pool_capacity(3, 1.5, false), 4u);
    Wide H = 0;
    for (std::size_t M = 1; M < 10001; ++M) {
        H += Wide(1) / M;
        const Wide wide = Wide("1.5") * (2 * (M + 1) * H - 4 * M);

        const auto expected =
            M == 3 ? 4u : boost::multiprecision::ceil(wide).convert_to<std::size_t>();
        ASSERT_EQ(dc_pool_capacity(M, 1.5, false), expected) << M;
        ASSERT_EQ(dc_pool_capacity(M, 1.5, true), expected) << M;
    }
}

TEST(V9, PaperDefaultsAndBeforeAllocationGuards) {
    InvestigationConfig c;
    c.rows = {0, 0, 0};
    c.mode = "band";
    EXPECT_EQ(c.bt, 29u);
    EXPECT_EQ(c.kt, 32u);
    EXPECT_EQ(c.ka, 49u);
    c.q = 128;
    c.p = 127;
    c.bq = 8;
    EXPECT_NO_THROW(c.validate());
    c.algorithm = "dc";
    c.kin = c.kout = 699050;
    EXPECT_NO_THROW(c.validate());
    c.kin = c.kout = 699051;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c.kin = SIZE_MAX;
    c.kout = SIZE_MAX;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c.kin = c.kout = 4;
    c.pattern = "P3";
    c.min_cluster = 1;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c.min_cluster = 2;
    EXPECT_NO_THROW(c.validate());
    c.windows.clear();
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c.windows = {{0, 1800}};
    c.pattern = "P4";
    c.q = 100;
    c.p = 5;
    c.bq = 7;
    c.kin = c.kout = 512;
    EXPECT_NO_THROW(c.validate());
    for (auto K : {1024u, 2048u, 4096u, 8192u, 10922u}) {
        c.kin = c.kout = K;
        EXPECT_NO_THROW(c.validate());
    }
    c.kin = c.kout = 10923;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c.kin = c.kout = 32768;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c.bt = 1;
    c.ba = 1;
    c.bq = 1;
    c.p = 0;
    c.q = 1;
    c.windows = {{0, 1}};
    c.kin = c.kout = 1u << 20;
    EXPECT_NO_THROW(c.validate());
}

TEST(V9, EveryAuxiliarySortHasAnIndependentExpectedBudget) {
    for (auto pattern : {"P1", "P2", "P3", "P4"}) {
        InvestigationConfig c;
        c.rows = {0, 0, 0};
        c.mode = "band";
        c.algorithm = "dc";
        c.pattern = pattern;
        c.kin = c.kout = 8;
        const auto p = compile_investigation(c);
        std::size_t sorts = 0;
        for (const auto &cmp : p.comparisons) {
            if (cmp.name.starts_with("dc.sort.") || cmp.name.ends_with(".compare")) {
                EXPECT_EQ(cmp.count, dc_pool_capacity(24, 1.5, false));
                ++sorts;
            }
        }
        EXPECT_EQ(sorts,
                  std::string(pattern) == "P1"   ? 3u
                  : std::string(pattern) == "P2" ? 5u
                  : std::string(pattern) == "P3" ? 2u
                                                 : 4u);
    }
}

TEST(V9, PublicP3BatchMasksAndDomainGuards) {
    InvestigationConfig c;
    c.rows = {0, 0, 0};
    c.mode = "band";
    c.pattern = "P3";
    c.kin = 7;
    c.kout = 9;
    c.p3_query_begin = 5;
    c.p3_query_count = 2;
    ASSERT_NO_THROW(c.validate());
    EXPECT_EQ(c.pairs(), 14u);
    auto copy = InvestigationConfig::parse(c.json());
    EXPECT_EQ(copy.p3_query_begin, 5u);
    EXPECT_EQ(copy.y_size(), 2u);
    const auto p = compile_investigation(c);
    const auto id = named_mask(p, "pair.time");
    const auto &terms = p.masks.at(id - p.inputs.size()).terms;
    ASSERT_EQ(terms.size(), 2u);
    EXPECT_EQ(terms[0].source_offset, 5u);
    EXPECT_EQ(terms[0].modulus, 2u);
    EXPECT_EQ(terms[1].divisor, 2u);
    EXPECT_EQ(terms[1].modulus, 7u);
    c.p3_query_begin = 6;
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c.p3_query_begin = 5;
    c.algorithm = "dc";
    EXPECT_THROW(c.validate(), std::invalid_argument);
    c.algorithm = "nl";
    c.pattern = "P1";
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(V9, PublicDimensionOverflowRejectedBeforeAllocation) {
    EXPECT_THROW(InvestigationConfig::parse(nlohmann::json{{"investigation", {{"n", 1025}}}}),
                 std::invalid_argument);
    InvestigationConfig c;
    c.rows = {SIZE_MAX, 1, 0};
    EXPECT_THROW(c.size(), std::invalid_argument);
    c.rows = {0, 0, 0};
    c.mode = "band";
    c.algorithm = "dc";
    c.pattern = "P4";
    c.kin = c.kout = 32768;
    c.ba = 62;
    c.bq = 30;
    c.bt = 1;
    c.q = 1;
    c.p = 0;
    c.windows = {{0, 1}};
    EXPECT_THROW(c.validate(), std::invalid_argument);
}
