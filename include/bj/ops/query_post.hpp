#pragma once
#include "bj/proto/and2a.hpp"
#include "bj/proto/bitmul.hpp"
#include "bj/proto/msb.hpp"

namespace bj {

    class QueryPost {
        InvestigationConfig c_;
        unsigned party_;
        std::span<const std::uint64_t> oi_, oo_;
        BitVec ni_{1}, no_{1}, selected_, alarms_{1}, weight_bits_;
        std::span<const std::uint64_t> weight_shares_;
        AlignedBuffer<std::uint64_t> nonoverflow_{1}, values_, count_{1}, alert64_{1};
        AlignedBuffer<std::uint32_t> query_selected_, diff_, alert32_{1};
        std::unique_ptr<FusedAnd2A<2, std::uint64_t>> omega_;
        std::unique_ptr<MsbFresh<std::uint32_t>> degree_, alarm32_;
        std::unique_ptr<MsbFresh<std::uint64_t>> alarm64_;
        std::unique_ptr<BitMul<std::uint64_t>> bitmul_;
        std::unique_ptr<FusedAnd2A<2, std::uint32_t>> query_filter_;
        std::vector<Step *> query_steps_;
        std::unique_ptr<RoundExecutor> query_round_;
        std::unique_ptr<Step> filter_;
        std::vector<Step *> open_, compare_, finish_, aopen_, acmp_;
        std::unique_ptr<RoundExecutor> oround_, cround_, fround_, aoround_, acround_;

    public:
        AlignedBuffer<std::uint64_t> degrees;
        QueryPost(const InvestigationConfig &,
                  MaterialStore &,
                  const MaterialPlan &,
                  const MaskedTable &y,
                  std::span<const std::uint64_t> oi,
                  std::span<const std::uint64_t> oo,
                  std::span<const std::uint64_t> amount_bits = {},
                  std::span<const std::uint64_t> valid_bits = {},
                  std::span<const std::uint64_t> query_filter = {},
                  std::span<const std::uint64_t> weight_shares = {});

        Step &omega_step() {
            return *omega_;
        }

        void prepare_omega();
        void aggregate(Net &);
        void alarm(Net &);

        std::span<std::uint64_t> count() {
            return count_.span();
        }

        std::span<const std::uint64_t> count() const {
            return count_.span();
        }

        std::uint64_t overflow() const {
            return (party_ == 0 ? 1ull : 0ull) - nonoverflow_.data()[0];
        }

        std::span<const std::uint64_t> alarms() const {
            return alarms_.words();
        }
    };
}
