#pragma once
#include "bj/plan/multi.hpp"
#include "bj/ref/band.hpp"

namespace bj::ref {
    struct FilteredRow {
        PlainRow row;
        std::uint16_t filters = 0;
    };

    struct TemplateResult {
        std::string name;
        uint128_t retained = 0, full = 0;
        bool retained_alarm = false, full_alarm = false;
    };

    struct ReferenceResult {
        bool overflow = false;
        std::size_t full_in = 0, full_out = 0, retained_in = 0, retained_out = 0;
        std::vector<TemplateResult> templates;
    };

    uint128_t aggregate(std::span<const PlainRow> inbound,
                        std::span<const PlainRow> outbound,
                        const InvestigationConfig &);
    ReferenceResult evaluate(const Plan &, std::span<const FilteredRow>, std::uint64_t subject);
    std::string decimal(uint128_t);
}
