#pragma once
#include "bj/core/multi_round.hpp"
#include "bj/core/stage_stats.hpp"
#include "bj/ops/compact.hpp"
#include "bj/plan/investigation.hpp"

namespace bj {
    struct QueryTemplate {
        std::string name, filter;
        InvestigationConfig config;
    };

    struct Plan {
        InvestigationConfig common;
        std::vector<QueryTemplate> templates;
        bool merge_rounds = true, share_thresholds = true, share_time_sorts = true;
        static Plan parse(const nlohmann::json &);
        static Plan load(const std::filesystem::path &);
        nlohmann::json json() const;
        void validate() const;
    };

    struct MultiCompilation {
        MaterialPlan material;
        std::vector<int> time_groups;
        std::vector<std::vector<std::size_t>> group_members;
        std::size_t shared_threshold_tables = 0;
        std::vector<std::vector<std::size_t>> interval_members;
        std::vector<std::vector<BandWindow>> interval_windows;
    };

    MultiCompilation compile_multi(const Plan &);
    MaterialPlan template_material_view(const MaterialPlan &, std::size_t);

    class MultiQuery {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        MultiQuery(const Plan &, MaterialStore &, const MaterialPlan &, const InputManager &);
        ~MultiQuery();
        void prepare(Net &, unsigned threads);
        void run(Net &);
        const MergeStats &stats() const;
        std::span<const std::uint64_t> counts(std::size_t) const;
        std::span<const std::uint64_t> alarms(std::size_t) const;
        std::uint64_t overflow() const;
        const CompactStats &compact_stats() const;
        const StageStats &union_stage() const;
        const StageStats &compact_stage() const;
        const StageStats &operator_stage() const;
        double template_seconds(std::size_t) const;
    };

    struct NetworkModel {
        double one_way_seconds = .0001, bytes_per_second = 125000000;
    };

    nlohmann::json
    enumerate_operators(const Plan &, const NetworkModel &, std::uint64_t budget_bits);
}
