#include "bj/ops/nl.hpp"
#include "bj/ops/patterns.hpp"
#include "bj/common/timer.hpp"

namespace bj {
    NestedLoop::NestedLoop(const InvestigationConfig &c,
                           MaterialStore &s,
                           const MaterialPlan &p,
                           const MaskedTable &in,
                           const MaskedTable &out,
                           std::span<const std::uint64_t> oi,
                           std::span<const std::uint64_t> oo,
                           const std::string &omega_gate,
                           std::span<const std::uint64_t> extra)
        : config_(c), party_(s.party()),
          band_(c.pattern == "P1" ? make_band_pair(c, s, p, in, out, extra) : nullptr),
          omega_in_(oi), omega_out_(oo), alarms_(c.windows.size()), count_(c.windows.size()),
          alert_difference_(c.windows.size()) {
        if (c.pattern != "P1") {
            pattern_ = make_pattern_operator(c, s, p, in, out, oi, oo, extra);
            return;
        }
        if (oi.size() != 1 || oo.size() != 1) {
            throw std::invalid_argument("overflow scalar share shape");
        }
        omega_step_ = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
            s,
            p,
            named_gate(p, omega_gate),
            std::array<std::span<const std::uint64_t>, 2>{neg_in_.words(), neg_out_.words()},
            nonoverflow_.span());
        if (!band_->opening_steps().empty()) {
            layers_.emplace_back(band_->opening_steps().begin(), band_->opening_steps().end());
        }
        layers_.emplace_back(band_->comparison_steps().begin(), band_->comparison_steps().end());
        layers_.emplace_back(band_->product_steps().begin(), band_->product_steps().end());
        layers_.back().push_back(omega_step_.get());
        for (auto &layer : layers_) {
            rounds_.push_back(std::make_unique<RoundExecutor>(layer));
        }
        if (c.output == "alarm") {
            alert_ = std::make_unique<MsbFresh<std::uint32_t>>(
                s, p, named_comparison(p, "band.alarm"), alert_difference_.span(), alarms_.words());
            alert_opening_[0] = &alert_->opening();
            alert_comparison_[0] = &alert_->comparison();
            alert_open_round_ = std::make_unique<RoundExecutor>(alert_opening_);
            alert_compare_round_ = std::make_unique<RoundExecutor>(alert_comparison_);
        }
    }

    void NestedLoop::run(Net &net) {
        if (pattern_) {
            pattern_->run(net);
            const auto &s = pattern_->stats();
            stats_ = {s.seconds, s.rounds, s.hops, s.contribution_bits};
            return;
        }
        if (used_) {
            throw std::logic_error("NL already executed");
        }
        used_ = true;
        const auto before = net.open_stats();
        Timer timer;
        band_->prepare();
        neg_in_.words()[0] = omega_in_[0] ^ (party_ == 0 ? 1u : 0u);
        neg_out_.words()[0] = omega_out_[0] ^ (party_ == 0 ? 1u : 0u);
        for (std::size_t i = 0; i < rounds_.size(); ++i) {
            if (i + 1 == rounds_.size()) {
                band_->combine();
            }
            rounds_[i]->run(net);
        }
        for (std::size_t w = 0; w < config_.windows.size(); ++w) {
            std::uint64_t sum = 0;
            const auto values = band_->shares(w);
#pragma omp parallel for reduction(+ : sum) if (values.size() >= 16384) schedule(static)
            for (std::size_t i = 0; i < values.size(); ++i) {
                sum += values[i];
            }
            count_.data()[w] = sum;
        }
        if (alert_) {
            for (std::size_t w = 0; w < config_.windows.size(); ++w) {
                alert_difference_.data()[w] =
                    static_cast<std::uint32_t>(count_.data()[w]) -
                    (party_ == 0 ? static_cast<std::uint32_t>(config_.threshold) : 0);
            }
            alert_open_round_->run(net);
            alert_compare_round_->run(net);
            if (party_ == 0) {
                for (std::size_t w = 0; w < alarms_.words().size(); ++w) {
                    alarms_.words()[w] ^= low_mask64(static_cast<unsigned>(
                        std::min<std::size_t>(64, config_.windows.size() - w * 64)));
                }
            }
        }
        const auto after = net.open_stats();
        stats_ = {timer.seconds(),
                  after.rounds - before.rounds,
                  after.hops - before.hops,
                  after.contribution_bits - before.contribution_bits};
    }

    NestedLoop::~NestedLoop() = default;

    void NestedLoop::evaluate_local() {
        if (!band_) {
            throw std::logic_error("local-only pattern aggregation benchmark is unavailable");
        }
        band_->evaluate_local();
    }

    std::span<const std::uint64_t> NestedLoop::count_shares() const {
        return pattern_ ? pattern_->count_shares() : count_.span();
    }

    std::uint64_t NestedLoop::overflow_share() const {
        return pattern_ ? pattern_->overflow_share()
                        : (party_ == 0 ? 1u : 0u) - nonoverflow_.data()[0];
    }

    std::span<const std::uint64_t> NestedLoop::alarm_shares() const {
        return pattern_ ? pattern_->alarm_shares() : alarms_.words();
    }
}
