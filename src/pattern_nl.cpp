#include "bj/common/timer.hpp"
#include "bj/ops/patterns.hpp"
#include "bj/ops/query_post.hpp"
#include "bj/proto/interval.hpp"

namespace bj {
    namespace {
        void slot_bits(const MaskedColumn &c, BitVec &bits, unsigned party) {
            for (std::size_t w = 0; w < bits.words().size(); ++w) {
                bits.words()[w] = c.mask_bits.words()[w] ^ (party == 0 ? c.bits.words()[w] : 0);
            }
        }

        class PatternNL final : public PatternOperator {
            InvestigationConfig c_;
            unsigned party_;
            const MaskedTable &in_;
            const MaskedTable &out_;
            BitVec amount_, valid_, query_amount_, query_valid_, tx_, ax_, vx_, anchor_;
            std::span<const std::uint64_t> extra_;
            AlignedBuffer<std::uint64_t> amount_delta_, time_delta_, products_;
            std::unique_ptr<BandPair> band_;
            std::unique_ptr<Interval<std::uint64_t>> amount_step_, time_step_;
            std::unique_ptr<Step> pair_step_;
            std::unique_ptr<QueryPost> post_;
            std::vector<std::vector<Step *>> layers_;
            std::vector<std::unique_ptr<RoundExecutor>> rounds_;
            DCStats stats_;
            bool used_ = false;

        public:
            PatternNL(const InvestigationConfig &c,
                      MaterialStore &s,
                      const MaterialPlan &p,
                      const MaskedTable &in,
                      const MaskedTable &out,
                      std::span<const std::uint64_t> oi,
                      std::span<const std::uint64_t> oo,
                      std::span<const std::uint64_t> extra)
                : c_(c), party_(s.party()), in_(in), out_(out), amount_(c.kin), valid_(c.kin),
                  query_amount_(c.y_size()), query_valid_(c.y_size()),
                  tx_(c.pattern == "P3" ? c.pairs() : 0), ax_(tx_.size()), vx_(tx_.size()),
                  anchor_(c.pattern == "P3" && !extra.empty() ? c.kin : 0), extra_(extra),
                  amount_delta_(c.kin), time_delta_(tx_.size()), products_(tx_.size()) {
                const auto &y = c.pattern == "P2" ? out : in;
                post_ = std::make_unique<QueryPost>(
                    c,
                    s,
                    p,
                    y,
                    oi,
                    oo,
                    c.pattern == "P3" ? query_amount_.words() : amount_.words(),
                    c.pattern == "P3" ? query_valid_.words() : valid_.words(),
                    anchor_.words());
                if (c.pattern == "P3") {
                    std::array<BandWindow, 1> aw{{{c.amount_lower, c.amount_upper - 1}}};
                    std::array<std::span<std::uint64_t>, 1> ao{amount_.words()}, to{tx_.words()};
                    amount_step_ = std::make_unique<Interval<std::uint64_t>>(
                        s,
                        p,
                        named_comparison(p, "pattern.amount.interval"),
                        amount_delta_.span(),
                        aw,
                        ao);
                    time_step_ = std::make_unique<Interval<std::uint64_t>>(
                        s,
                        p,
                        named_comparison(p, "pattern.time.interval"),
                        time_delta_.span(),
                        c.windows,
                        to);
                    if (p.gates[named_gate(p, "pattern.pair")].m == 3) {
                        pair_step_ = std::make_unique<FusedAnd2A<3, std::uint64_t>>(
                            s,
                            p,
                            named_gate(p, "pattern.pair"),
                            std::array<std::span<const std::uint64_t>, 3>{
                                tx_.words(), ax_.words(), vx_.words()},
                            products_.span());
                    } else {
                        pair_step_ = std::make_unique<FusedAnd2A<4, std::uint64_t>>(
                            s,
                            p,
                            named_gate(p, "pattern.pair"),
                            std::array<std::span<const std::uint64_t>, 4>{
                                tx_.words(), ax_.words(), vx_.words(), extra},
                            products_.span());
                    }
                    layers_ = {{amount_step_.get(), time_step_.get()},
                               {pair_step_.get(), &post_->omega_step()}};
                } else {
                    auto role = c;
                    if (c.pattern == "P4") {
                        role.kin = c.kout;
                        role.kout = c.kin;
                        role.pattern = "P4";
                    }
                    band_ = make_band_pair(role,
                                           s,
                                           p,
                                           c.pattern == "P4" ? out : in,
                                           c.pattern == "P4" ? in : out,
                                           extra);
                    if (!band_->opening_steps().empty()) {
                        layers_.emplace_back(band_->opening_steps().begin(),
                                             band_->opening_steps().end());
                    }
                    layers_.emplace_back(band_->comparison_steps().begin(),
                                         band_->comparison_steps().end());
                    layers_.emplace_back(band_->product_steps().begin(),
                                         band_->product_steps().end());
                    layers_.back().push_back(&post_->omega_step());
                }
                for (auto &l : layers_) {
                    rounds_.push_back(std::make_unique<RoundExecutor>(l));
                }
            }

            void run(Net &net) override {
                if (used_) {
                    throw std::logic_error("pattern NL reused");
                }
                used_ = true;
                const auto before = net.open_stats();
                Timer timer;
                post_->prepare_omega();
                for (std::size_t y = 0; y < anchor_.size(); ++y) {
                    const auto index = y * c_.kin + y;
                    anchor_.set(y, ((extra_[index / 64] >> (index % 64)) & 1) != 0);
                }
                if (band_) {
                    band_->prepare();
                } else {
                    slot_bits(in_.columns[2], valid_, party_);
                    for (std::size_t i = 0; i < c_.kin; ++i) {
                        amount_delta_.data()[i] =
                            static_cast<std::uint64_t>(in_.columns[1].d128.data()[i]);
                    }
                    for (std::size_t i = 0; i < c_.pairs(); ++i) {
                        time_delta_.data()[i] =
                            (in_.columns[0].d64.data()[c_.p3_query_begin + i % c_.y_size()] -
                             in_.columns[0].d64.data()[i / c_.y_size()]) &
                            width_mask(c_.kt);
                    }
                }
                for (std::size_t layer = 0; layer < rounds_.size(); ++layer) {
                    if (layer + 1 == rounds_.size()) {
                        if (band_) {
                            band_->combine();
                        } else {
                            for (std::size_t i = 0; i < c_.pairs(); ++i) {
                                const auto x = i / c_.y_size(),
                                           y = c_.p3_query_begin + i % c_.y_size();
                                if (x == y) {
                                    tx_.set(i, false);
                                }
                                ax_.set(i, amount_.get(x));
                                vx_.set(i, valid_.get(x));
                            }
                        }
                    }
                    rounds_[layer]->run(net);
                }
                if (c_.pattern == "P3") {
                    for (std::size_t y = 0; y < c_.y_size(); ++y) {
                        query_amount_.set(y, amount_.get(c_.p3_query_begin + y));
                        query_valid_.set(y, valid_.get(c_.p3_query_begin + y));
                    }
                }
                if (c_.pattern == "P4") {
                    std::uint64_t count = 0;
                    for (auto v : band_->shares(0)) {
                        count += v;
                    }
                    post_->count()[0] = count;
                } else {
                    auto degrees = post_->degrees.span();
                    std::fill(degrees.begin(), degrees.end(), 0);
                    auto product = band_ ? band_->shares(0) : products_.span();
                    for (std::size_t x = 0; x < c_.x_size(); ++x) {
                        for (std::size_t y = 0; y < c_.y_size(); ++y) {
                            degrees[y] += product[x * c_.y_size() + y];
                        }
                    }
                    post_->aggregate(net);
                }
                post_->alarm(net);
                const auto after = net.open_stats();
                stats_.seconds = timer.seconds();
                stats_.rounds = after.rounds - before.rounds;
                stats_.hops = after.hops - before.hops;
                stats_.contribution_bits = after.contribution_bits - before.contribution_bits;
            }

            std::span<const std::uint64_t> count_shares() const override {
                return post_->count();
            }

            std::uint64_t overflow_share() const override {
                return post_->overflow();
            }

            std::span<const std::uint64_t> alarm_shares() const override {
                return post_->alarms();
            }

            const DCStats &stats() const override {
                return stats_;
            }
        };
    }

    std::unique_ptr<PatternOperator> make_pattern_nl(const InvestigationConfig &c,
                                                     MaterialStore &s,
                                                     const MaterialPlan &p,
                                                     const MaskedTable &in,
                                                     const MaskedTable &out,
                                                     std::span<const std::uint64_t> oi,
                                                     std::span<const std::uint64_t> oo,
                                                     std::span<const std::uint64_t> extra) {
        return std::make_unique<PatternNL>(c, s, p, in, out, oi, oo, extra);
    }
}
