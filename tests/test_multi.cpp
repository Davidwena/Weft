#include "bj/ops/quicksort.hpp"
#include "bj/plan/multi.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
using namespace bj;

namespace {
    nlohmann::json tiny() {
        return {{"multi",
                 {{"common",
                   {{"n", 2}, {"rows", {3, 3}}, {"kin", 3}, {"kout", 3}, {"windows", {{0, 4}}}}},
                  {"templates",
                   {{{"name", "count"}, {"pattern", "P1"}, {"window", {0, 4}}, {"operator", "nl"}},
                    {{"name", "amount"},
                     {"pattern", "P2"},
                     {"window", {0, 3}},
                     {"operator", "nl"}}}}}}};
    }
}

TEST(Multi, SharedThresholdEndpointsAndPhysicalReuseGuard) {
    const auto plan = Plan::parse(tiny());
    const auto compiled = compile_multi(plan);
    EXPECT_EQ(compiled.shared_threshold_tables, 1u);
    const auto &p = compiled.material;
    MaterialLayout layout(p, 0, MaterialMode::Random);
    const auto a = named_comparison(p, "q0.band.interval"),
               b = named_comparison(p, "q1.band.interval");
    EXPECT_EQ(layout.comparisons[a].tables, layout.comparisons[b].tables);
    EXPECT_EQ(p.comparisons[a].evaluations, 3u);
    ASSERT_FALSE(p.comparisons[b].evaluation_sources.empty());
    EXPECT_EQ(layout.comparisons[a].evaluations[p.comparisons[b].evaluation_sources[0]],
              layout.comparisons[b].evaluations[0]);
    EXPECT_NE(layout.comparisons[a].evaluations[0], layout.comparisons[a].evaluations[1]);

    struct Temp {
        std::filesystem::path path;

        Temp() {
            char name[] = "/tmp/bj-t7-XXXXXX";
            auto *d = ::mkdtemp(name);
            if (!d) {
                throw std::runtime_error("temporary test directory");
            }
            path = d;
        }

        ~Temp() {
            std::error_code e;
            std::filesystem::remove_all(path, e);
        }
    } tmp;

    generate_materials(p, MaterialMode::Prg, tmp.path / "material");
    MaterialStore store(p, 0, tmp.path / "material/party_0.bin");
    store.claim_execution();
    store.claim_comparison(a, p.comparisons[b].evaluation_sources[0]);
    EXPECT_THROW(store.claim_comparison(b, 0), std::logic_error);
    auto bad = p;
    bad.comparisons[b].inputs = bad.comparisons[named_comparison(p, "q1.band.amount1")].inputs;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = p;
    bad.comparisons[b].table_source = bad.comparisons[b].name;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = p;
    bad.comparisons[b].kind = CompareKind::MsbFresh;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    auto standalone = plan;
    standalone.share_thresholds = false;
    const auto independent = compile_multi(standalone);
    MaterialLayout full(independent.material, 0, MaterialMode::Random);
    EXPECT_LT(layout.logical_share_bits, full.logical_share_bits);
    EXPECT_EQ(Plan::parse(plan.json()).json(), plan.json());
}

TEST(Multi, SharedTimeSortCarriesAllPayloadsAndBudgetEnumeratesEveryChoice) {
    auto plan = Plan::parse(tiny());
    for (auto &q : plan.templates) {
        q.config.algorithm = "dc";
        q.config.windows = {{0, 4}};
    }
    const auto compiled = compile_multi(plan);
    ASSERT_EQ(compiled.group_members.size(), 1u);
    EXPECT_EQ(compiled.group_members[0], (std::vector<std::size_t>{0, 1}));
    const auto &p = compiled.material;
    const auto cmp = named_comparison(p, "multi.time.0.compare");
    EXPECT_EQ(p.comparisons[cmp].width, 64u);
    EXPECT_EQ(std::count_if(p.shuffles.begin(),
                            p.shuffles.end(),
                            [](auto &s) {
                                return s.name == "multi.time.0";
                            }),
              1);
    auto mixed_policy = plan;
    mixed_policy.common.kin = mixed_policy.common.kout = 32;
    for (auto &q : mixed_policy.templates) {
        q.config.kin = q.config.kout = 32;
    }
    mixed_policy.templates[0].config.dc_pool_safe = false;
    mixed_policy.templates[0].config.dc_pool_factor = .01;
    const auto safe_compilation = compile_multi(mixed_policy);
    EXPECT_GE(safe_compilation.material
                  .comparisons[named_comparison(safe_compilation.material, "multi.time.0.compare")]
                  .count,
              dc_pool_capacity(96, 1.5, true));
    const auto model = enumerate_operators(plan, {.001, 125000000}, UINT64_MAX);
    ASSERT_EQ(model["candidates"].size(), 4u);
    ASSERT_FALSE(model["selected_mask"].is_null());
    const auto chosen = model["selected_mask"].get<std::size_t>();
    EXPECT_TRUE(model["candidates"][chosen]["feasible"].get<bool>());
    const auto exact_budget =
        model["candidates"][chosen]["preprocessing_bits"].get<std::uint64_t>();
    auto limited = enumerate_operators(plan, {.001, 125000000}, exact_budget);
    for (const auto &row : limited["candidates"]) {
        if (row["feasible"].get<bool>()) {
            EXPECT_LE(row["preprocessing_bits"].get<std::uint64_t>(), exact_budget);
        }
    }
    EXPECT_TRUE(enumerate_operators(plan, {.001, 125000000}, 0)["selected_mask"].is_null());
    EXPECT_THROW(enumerate_operators(plan, {0, 0}, UINT64_MAX), std::invalid_argument);
}

TEST(Multi, ThreeWindowsMatchThePaperEndpointPrice) {
    auto doc = tiny();
    doc["multi"]["templates"] = nlohmann::json::array();
    for (unsigned j = 0; j < 3; ++j) {
        doc["multi"]["templates"].push_back({{"name", "w" + std::to_string(j)},
                                             {"pattern", "P1"},
                                             {"window", {0, 2 + 2 * j}},
                                             {"operator", "nl"}});
    }
    const auto plan = Plan::parse(doc);
    const auto compilation = compile_multi(plan);
    ASSERT_EQ(compilation.interval_members.size(), 1u);
    const auto &p = compilation.material;
    const auto id = named_comparison(p, "q0.band.interval");
    EXPECT_EQ(p.comparisons[id].evaluations, 4u);
    MaterialLayout layout(p, 0, MaterialMode::Random);
    std::uint64_t bits = 0;
    for (auto seg : layout.comparisons[id].tables) {
        bits += layout.segments[seg].bits;
    }
    for (const auto &e : layout.comparisons[id].evaluations) {
        for (auto seg : e) {
            bits += layout.segments[seg].bits;
        }
    }
    EXPECT_EQ(bits, 984 * plan.common.pairs());
}
