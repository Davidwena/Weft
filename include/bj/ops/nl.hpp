#pragma once
#include "bj/proto/bandpair.hpp"

namespace bj {
    class PatternOperator;

    struct NLStats {
        double seconds = 0;
        std::uint64_t rounds = 0, hops = 0, contribution_bits = 0;
    };

    class NestedLoop {
        std::unique_ptr<PatternOperator> pattern_;
        InvestigationConfig config_;
        unsigned party_;
        std::unique_ptr<BandPair> band_;
        std::span<const std::uint64_t> omega_in_, omega_out_;
        BitVec neg_in_{1}, neg_out_{1}, alarms_;
        AlignedBuffer<std::uint64_t> nonoverflow_{1}, count_;
        AlignedBuffer<std::uint32_t> alert_difference_;
        std::unique_ptr<FusedAnd2A<2, std::uint64_t>> omega_step_;
        std::unique_ptr<MsbFresh<std::uint32_t>> alert_;
        std::vector<std::vector<Step *>> layers_;
        std::vector<std::unique_ptr<RoundExecutor>> rounds_;
        std::array<Step *, 1> alert_opening_{}, alert_comparison_{};
        std::unique_ptr<RoundExecutor> alert_open_round_, alert_compare_round_;
        NLStats stats_;
        bool used_ = false;

    public:
        NestedLoop(const InvestigationConfig &,
                   MaterialStore &,
                   const MaterialPlan &,
                   const MaskedTable &,
                   const MaskedTable &,
                   std::span<const std::uint64_t> omega_in,
                   std::span<const std::uint64_t> omega_out,
                   const std::string &omega_gate = "output.omega",
                   std::span<const std::uint64_t> extra = {});
        ~NestedLoop();
        void run(Net &);
        void evaluate_local();
        std::span<const std::uint64_t> count_shares() const;
        std::uint64_t overflow_share() const;
        std::span<const std::uint64_t> alarm_shares() const;

        const NLStats &stats() const {
            return stats_;
        }
    };
}
