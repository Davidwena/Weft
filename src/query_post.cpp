#include "bj/ops/query_post.hpp"

namespace bj {
    QueryPost::QueryPost(const InvestigationConfig &c,
                         MaterialStore &s,
                         const MaterialPlan &p,
                         const MaskedTable &y,
                         std::span<const std::uint64_t> oi,
                         std::span<const std::uint64_t> oo,
                         std::span<const std::uint64_t> amount_bits,
                         std::span<const std::uint64_t> valid_bits,
                         std::span<const std::uint64_t> query_filter,
                         std::span<const std::uint64_t> weight_shares)
        : c_(c), party_(s.party()), oi_(oi), oo_(oo), selected_(c.y_size()),
          weight_bits_(weight_shares.empty() ? 0 : c.y_size()), weight_shares_(weight_shares),
          values_(c.y_size()), query_selected_(query_filter.empty() ? 0 : c.y_size()),
          diff_(c.y_size()), degrees(c.y_size()) {
        std::fill(degrees.span().begin(), degrees.span().end(), 0);
        count_.data()[0] = 0;
        if (oi.size() != 1 || oo.size() != 1) {
            throw std::invalid_argument("overflow share shape");
        }
        omega_ = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
            s,
            p,
            named_gate(p, "output.omega"),
            std::array<std::span<const std::uint64_t>, 2>{ni_.words(), no_.words()},
            nonoverflow_.span());
        if (c.pattern == "P2" || c.pattern == "P3") {
            degree_ = std::make_unique<MsbFresh<std::uint32_t>>(
                s, p, named_comparison(p, "pattern.degree"), diff_.span(), selected_.words());
            open_ = {&degree_->opening()};
            compare_ = {&degree_->comparison()};
            oround_ = std::make_unique<RoundExecutor>(open_);
            cround_ = std::make_unique<RoundExecutor>(compare_);
            if (c.pattern == "P2") {
                bitmul_ = std::make_unique<BitMul<std::uint64_t>>(s,
                                                                  p,
                                                                  named_bitmul(p, "pattern.amount"),
                                                                  selected_.words(),
                                                                  y.columns[1],
                                                                  values_.span());
                finish_ = {bitmul_.get()};
            } else {
                if (p.gates[named_gate(p, "pattern.filter")].m == 2) {
                    if (weight_shares.size() != c.y_size()) {
                        throw std::invalid_argument("P3 DC requires arithmetic weight shares");
                    }
                    filter_ = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
                        s,
                        p,
                        named_gate(p, "pattern.filter"),
                        std::array<std::span<const std::uint64_t>, 2>{selected_.words(),
                                                                      weight_bits_.words()},
                        values_.span());
                } else if (p.gates[named_gate(p, "pattern.filter")].m == 3) {
                    filter_ = std::make_unique<FusedAnd2A<3, std::uint64_t>>(
                        s,
                        p,
                        named_gate(p, "pattern.filter"),
                        std::array<std::span<const std::uint64_t>, 3>{
                            selected_.words(), amount_bits, valid_bits},
                        values_.span());
                } else {
                    filter_ = std::make_unique<FusedAnd2A<4, std::uint64_t>>(
                        s,
                        p,
                        named_gate(p, "pattern.filter"),
                        std::array<std::span<const std::uint64_t>, 4>{
                            selected_.words(), amount_bits, valid_bits, query_filter},
                        values_.span());
                }
                finish_ = {filter_.get()};
            }
            fround_ = std::make_unique<RoundExecutor>(finish_);
        }
        if (!query_filter.empty() && c.pattern != "P3") {
            if (c.pattern != "P2") {
                throw std::invalid_argument("query filter is only for P2 degree aggregation");
            }
            query_filter_ = std::make_unique<FusedAnd2A<2, std::uint32_t>>(
                s,
                p,
                named_gate(p, "pattern.query_filter"),
                std::array<std::span<const std::uint64_t>, 2>{selected_.words(), query_filter},
                query_selected_.span());
            query_steps_ = {query_filter_.get()};
            query_round_ = std::make_unique<RoundExecutor>(query_steps_);
        }
        if (c.output == "alarm") {
            if (c.pattern == "P2") {
                alarm64_ = std::make_unique<MsbFresh<std::uint64_t>>(
                    s, p, named_comparison(p, "pattern.alarm"), alert64_.span(), alarms_.words());
                aopen_ = {&alarm64_->opening()};
                acmp_ = {&alarm64_->comparison()};
            } else {
                alarm32_ = std::make_unique<MsbFresh<std::uint32_t>>(
                    s, p, named_comparison(p, "pattern.alarm"), alert32_.span(), alarms_.words());
                aopen_ = {&alarm32_->opening()};
                acmp_ = {&alarm32_->comparison()};
            }
            aoround_ = std::make_unique<RoundExecutor>(aopen_);
            acround_ = std::make_unique<RoundExecutor>(acmp_);
        }
    }

    void QueryPost::prepare_omega() {
        ni_.words()[0] = oi_[0] ^ (party_ == 0 ? 1u : 0u);
        no_.words()[0] = (c_.pattern == "P3" ? 0 : oo_[0]) ^ (party_ == 0 ? 1u : 0u);
    }

    void QueryPost::aggregate(Net &net) {
        if (!degree_) {
            return;
        }
        for (std::size_t i = 0; i < degrees.size(); ++i) {
            diff_.data()[i] =
                static_cast<std::uint32_t>(degrees.data()[i]) -
                (party_ == 0
                     ? static_cast<std::uint32_t>(c_.pattern == "P2" ? 1 : c_.min_cluster - 1)
                     : 0u);
        }
        oround_->run(net);
        cround_->run(net);
        if (party_ == 0) {
            for (std::size_t w = 0; w < selected_.words().size(); ++w) {
                selected_.words()[w] ^= low_mask64(
                    static_cast<unsigned>(std::min<std::size_t>(64, degrees.size() - w * 64)));
            }
        }
        if (query_round_) {
            query_round_->run(net);
            for (std::size_t i = 0; i < degrees.size(); ++i) {
                selected_.set(i, query_selected_.data()[i] & 1);
            }
        }

        for (std::size_t i = 0; i < weight_bits_.size(); ++i) {
            weight_bits_.set(i, (weight_shares_[i] & 1) != 0);
        }
        fround_->run(net);
        std::uint64_t sum = 0;
        for (auto x : values_.span()) {
            sum += x;
        }
        count_.data()[0] = sum;
    }

    void QueryPost::alarm(Net &net) {
        if (!aoround_) {
            return;
        }
        alert64_.data()[0] = count_.data()[0] - (party_ == 0 ? c_.threshold : 0u);
        alert32_.data()[0] = static_cast<std::uint32_t>(alert64_.data()[0]);
        aoround_->run(net);
        acround_->run(net);
        if (party_ == 0) {
            alarms_.words()[0] ^= 1;
        }
    }
}
