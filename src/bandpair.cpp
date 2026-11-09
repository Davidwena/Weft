#include "bj/proto/bandpair.hpp"
#include "bj/proto/eq.hpp"

namespace bj {
    namespace {
        template <class TT, class AT> class NativeBandPair final : public BandPair {
            InvestigationConfig c_;
            unsigned party_;
            const MaskedTable &in_;
            const MaskedTable &out_;
            AlignedBuffer<TT> time_, time_alt_, time_shares_, time_alt_shares_, fresh_time_,
                fresh_hat_;
            AlignedBuffer<AT> amount1_, amount2_, share1_, share2_;
            AlignedBuffer<std::uint64_t> cp_delta_;
            BitVec cp_equal_;
            std::unique_ptr<EqComparison<std::uint64_t>> cp_step_;
            BitVec t0_, t1_, a0_, a1_;
            std::vector<BitVec> interval_bits_, time_good_, amount_good_;
            std::vector<AlignedBuffer<std::uint64_t>> products_;
            std::unique_ptr<Interval<TT>> interval_;
            std::unique_ptr<MaskedRingOpen<TT>> interval_open_;
            std::unique_ptr<MsbKnown<TT>> tknown0_, tknown1_;
            std::unique_ptr<MsbFresh<TT>> tfresh0_, tfresh1_;
            std::unique_ptr<MsbKnown<AT>> aknown0_, aknown1_;
            std::unique_ptr<MsbFresh<AT>> afresh0_, afresh1_;
            std::vector<std::unique_ptr<Step>> gates_;
            std::vector<unsigned> arities_;
            std::vector<Step *> opening_, comparison_, product_;

        public:
            NativeBandPair(const InvestigationConfig &c,
                           MaterialStore &s,
                           const MaterialPlan &p,
                           const MaskedTable &inbound,
                           const MaskedTable &outbound,
                           std::span<const std::uint64_t> extra)
                : c_(c), party_(s.party()), in_(inbound), out_(outbound), time_(c.pairs()),
                  time_alt_(c.pairs()), time_shares_(c.pairs()), time_alt_shares_(c.pairs()),
                  fresh_time_(c.interval() && !c.zero_open() ? c.pairs() : 0),
                  fresh_hat_(fresh_time_.size()), amount1_(c.pairs()), amount2_(c.pairs()),
                  share1_(c.pairs()), share2_(c.pairs()),
                  cp_delta_(c.pattern == "P4" ? c.pairs() : 0),
                  cp_equal_(c.pattern == "P4" ? c.pairs() : 0), t0_(c.pairs()), t1_(c.pairs()),
                  a0_(c.pairs()), a1_(c.pairs()) {
                const auto P = c.pairs();
                interval_bits_.reserve(c.windows.size());
                time_good_.reserve(c.windows.size());
                amount_good_.reserve(c.windows.size());
                products_.reserve(c.windows.size());
                for (std::size_t w = 0; w < c.windows.size(); ++w) {
                    interval_bits_.emplace_back(P);
                    time_good_.emplace_back(P);
                    amount_good_.emplace_back(P);
                    products_.emplace_back(P);
                }
                if (c.interval()) {
                    const auto id = named_comparison(p, "band.interval");
                    std::span<const TT> masked = time_.span();
                    if (!c.zero_open()) {
                        unpack_native<TT>(
                            c.kt, s.packed(s.layout().comparisons[id].aux[0]), fresh_time_.span());
                        interval_open_ = std::make_unique<MaskedRingOpen<TT>>(
                            c.kt, time_shares_.span(), fresh_time_.span(), fresh_hat_.span());
                        masked = fresh_hat_.span();
                        opening_.push_back(interval_open_.get());
                    }
                    std::vector<std::span<std::uint64_t>> outputs;
                    for (auto &bits : interval_bits_) {
                        outputs.push_back(bits.words());
                    }
                    interval_ =
                        std::make_unique<Interval<TT>>(s, p, id, masked, c.windows, outputs);
                    comparison_.push_back(interval_.get());
                } else if (c.zero_open()) {
                    tknown0_ = std::make_unique<MsbKnown<TT>>(
                        s, p, named_comparison(p, "band.time0"), 0, time_.span(), t0_.words());
                    tknown1_ = std::make_unique<MsbKnown<TT>>(
                        s, p, named_comparison(p, "band.time1"), 0, time_alt_.span(), t1_.words());
                    comparison_.push_back(tknown0_.get());
                    comparison_.push_back(tknown1_.get());
                } else {
                    tfresh0_ = std::make_unique<MsbFresh<TT>>(
                        s, p, named_comparison(p, "band.time0"), time_shares_.span(), t0_.words());
                    tfresh1_ = std::make_unique<MsbFresh<TT>>(s,
                                                              p,
                                                              named_comparison(p, "band.time1"),
                                                              time_alt_shares_.span(),
                                                              t1_.words());
                    opening_.push_back(&tfresh0_->opening());
                    opening_.push_back(&tfresh1_->opening());
                    comparison_.push_back(&tfresh0_->comparison());
                    comparison_.push_back(&tfresh1_->comparison());
                }
                if (c.zero_open()) {
                    aknown0_ = std::make_unique<MsbKnown<AT>>(
                        s, p, named_comparison(p, "band.amount1"), 0, amount1_.span(), a0_.words());
                    aknown1_ = std::make_unique<MsbKnown<AT>>(
                        s, p, named_comparison(p, "band.amount2"), 0, amount2_.span(), a1_.words());
                    comparison_.push_back(aknown0_.get());
                    comparison_.push_back(aknown1_.get());
                } else {
                    afresh0_ = std::make_unique<MsbFresh<AT>>(
                        s, p, named_comparison(p, "band.amount1"), share1_.span(), a0_.words());
                    afresh1_ = std::make_unique<MsbFresh<AT>>(
                        s, p, named_comparison(p, "band.amount2"), share2_.span(), a1_.words());
                    opening_.push_back(&afresh0_->opening());
                    opening_.push_back(&afresh1_->opening());
                    comparison_.push_back(&afresh0_->comparison());
                    comparison_.push_back(&afresh1_->comparison());
                }
                if (c.pattern == "P4") {
                    cp_step_ = std::make_unique<EqComparison<std::uint64_t>>(
                        s,
                        p,
                        named_comparison(p, "band.cp"),
                        0,
                        cp_delta_.span(),
                        cp_equal_.words());
                    comparison_.push_back(cp_step_.get());
                }
                for (std::size_t w = 0; w < c.windows.size(); ++w) {
                    const auto id = named_gate(p, "band.product." + std::to_string(w));
                    if ((!extra.empty() ||
                         (!P && c.interval() && c.amount() && p.gates[id].m == 4)) &&
                        c.pattern == "P4") {
                        gates_.push_back(std::make_unique<FusedAnd2A<4, std::uint64_t>>(
                            s,
                            p,
                            id,
                            std::array<std::span<const std::uint64_t>, 4>{interval_bits_[w].words(),
                                                                          amount_good_[w].words(),
                                                                          cp_equal_.words(),
                                                                          extra},
                            products_[w].span()));
                    } else if (!extra.empty() || (!P && c.pattern != "P4" && c.interval() &&
                                                  c.amount() && p.gates[id].m == 3)) {
                        gates_.push_back(std::make_unique<FusedAnd2A<3, std::uint64_t>>(
                            s,
                            p,
                            id,
                            std::array<std::span<const std::uint64_t>, 3>{
                                interval_bits_[w].words(), amount_good_[w].words(), extra},
                            products_[w].span()));
                    } else if (c.pattern == "P4") {
                        gates_.push_back(std::make_unique<FusedAnd2A<3, std::uint64_t>>(
                            s,
                            p,
                            id,
                            std::array<std::span<const std::uint64_t>, 3>{interval_bits_[w].words(),
                                                                          amount_good_[w].words(),
                                                                          cp_equal_.words()},
                            products_[w].span()));
                    } else if (c.interval() && c.amount()) {
                        gates_.push_back(std::make_unique<FusedAnd2A<2, std::uint64_t>>(
                            s,
                            p,
                            id,
                            std::array<std::span<const std::uint64_t>, 2>{interval_bits_[w].words(),
                                                                          amount_good_[w].words()},
                            products_[w].span()));
                    } else if (c.interval()) {
                        gates_.push_back(std::make_unique<FusedAnd2A<3, std::uint64_t>>(
                            s,
                            p,
                            id,
                            std::array<std::span<const std::uint64_t>, 3>{
                                interval_bits_[w].words(), a0_.words(), a1_.words()},
                            products_[w].span()));
                    } else if (c.amount()) {
                        gates_.push_back(std::make_unique<FusedAnd2A<3, std::uint64_t>>(
                            s,
                            p,
                            id,
                            std::array<std::span<const std::uint64_t>, 3>{
                                t0_.words(), t1_.words(), amount_good_[w].words()},
                            products_[w].span()));
                    } else {
                        gates_.push_back(std::make_unique<FusedAnd2A<4, std::uint64_t>>(
                            s,
                            p,
                            id,
                            std::array<std::span<const std::uint64_t>, 4>{
                                t0_.words(), t1_.words(), a0_.words(), a1_.words()},
                            products_[w].span()));
                    }
                    arities_.push_back(p.gates[id].m);
                    product_.push_back(gates_.back().get());
                }
            }

            void prepare() override {
                if (in_.columns.size() < 2 || out_.columns.size() < 2) {
                    throw std::invalid_argument("band needs time/amount slots");
                }
                const auto ti = in_.columns[0].d64.span(), to = out_.columns[0].d64.span(),
                           mi = in_.columns[0].m64.span(), mo = out_.columns[0].m64.span();
                const auto ai = in_.columns[1].d128.span(), ao = out_.columns[1].d128.span(),
                           ami = in_.columns[1].m128.span(), amo = out_.columns[1].m128.span();
                const auto maskt = native_mask<TT>(c_.kt), maska = native_mask<AT>(c_.ka);
                const auto L = static_cast<TT>(c_.windows[0].lower),
                           U = static_cast<TT>(c_.windows[0].upper + 1);
#pragma omp parallel for if (c_.pairs() >= 16384) schedule(static)
                for (std::size_t i = 0; i < c_.pairs(); ++i) {
                    const auto x = i / c_.kout, y = i % c_.kout;
                    const auto delta = static_cast<TT>(to[y] - ti[x]);
                    const auto lambda = static_cast<TT>(mo[y] - mi[x]);
                    if (c_.pattern == "P4") {
                        cp_delta_.data()[i] = static_cast<std::uint64_t>(
                            out_.columns[3].d128.data()[y] - in_.columns[3].d128.data()[x]);
                    }
                    time_.data()[i] = (delta - (c_.interval() ? TT{0} : L)) & maskt;
                    time_alt_.data()[i] = (delta - U) & maskt;
                    time_shares_.data()[i] = ((party_ == 0 ? delta : TT{0}) - lambda -
                                              (party_ == 0 && !c_.interval() ? L : TT{0})) &
                                             maskt;
                    time_alt_shares_.data()[i] =
                        ((party_ == 0 ? delta : TT{0}) - lambda - (party_ == 0 ? U : TT{0})) &
                        maskt;
                    const auto z1 = static_cast<AT>((c_.q + c_.p) * ai[x] - c_.q * ao[y]),
                               z2 = static_cast<AT>(c_.q * ao[y] - (c_.q - c_.p) * ai[x]);
                    const auto r1 = static_cast<AT>((c_.q + c_.p) * ami[x] - c_.q * amo[y]),
                               r2 = static_cast<AT>(c_.q * amo[y] - (c_.q - c_.p) * ami[x]);
                    amount1_.data()[i] = z1 & maska;
                    amount2_.data()[i] = z2 & maska;
                    share1_.data()[i] = ((party_ == 0 ? z1 : AT{0}) - r1) & maska;
                    share2_.data()[i] = ((party_ == 0 ? z2 : AT{0}) - r2) & maska;
                }
            }

            void combine() override {
                for (std::size_t word = 0; word < a0_.words().size(); ++word) {
                    const auto take =
                        static_cast<unsigned>(std::min<std::size_t>(64, c_.pairs() - word * 64));
                    const auto one = party_ == 0 ? low_mask64(take) : 0;
                    for (std::size_t w = 0; w < c_.windows.size(); ++w) {
                        amount_good_[w].words()[word] = one ^ a0_.words()[word] ^ a1_.words()[word];
                    }

                    if (!c_.interval()) {
                        t0_.words()[word] ^= one;
                    }
                    if (!c_.amount()) {
                        a0_.words()[word] ^= one;
                        a1_.words()[word] ^= one;
                    }
                }
            }

            void evaluate_local() override {
                if (interval_) {
                    interval_->evaluate_local();
                } else if (c_.zero_open()) {
                    tknown0_->evaluate_local(t0_.words());
                    tknown1_->evaluate_local(t1_.words());
                } else {
                    tfresh0_->evaluate_local(t0_.words());
                    tfresh1_->evaluate_local(t1_.words());
                }
                if (c_.zero_open()) {
                    aknown0_->evaluate_local(a0_.words());
                    aknown1_->evaluate_local(a1_.words());
                } else {
                    afresh0_->evaluate_local(a0_.words());
                    afresh1_->evaluate_local(a1_.words());
                }
                if (cp_step_) {
                    cp_step_->evaluate_local(cp_equal_.words());
                }
                combine();
                for (std::size_t i = 0; i < gates_.size(); ++i) {
                    auto &gate = gates_[i];
                    const auto m = arities_[i];
                    if (m == 2) {
                        static_cast<FusedAnd2A<2, std::uint64_t> &>(*gate).evaluate_local();
                    } else if (m == 3) {
                        static_cast<FusedAnd2A<3, std::uint64_t> &>(*gate).evaluate_local();
                    } else {
                        static_cast<FusedAnd2A<4, std::uint64_t> &>(*gate).evaluate_local();
                    }
                }
            }

            std::span<Step *const> opening_steps() override {
                return opening_;
            }

            std::span<Step *const> comparison_steps() override {
                return comparison_;
            }

            std::span<Step *const> product_steps() override {
                return product_;
            }

            std::span<const std::uint64_t> shares(std::size_t w) const override {
                return products_.at(w).span();
            }
        };
    }

    std::unique_ptr<BandPair> make_band_pair(const InvestigationConfig &c,
                                             MaterialStore &s,
                                             const MaterialPlan &p,
                                             const MaskedTable &in,
                                             const MaskedTable &out,
                                             std::span<const std::uint64_t> extra) {
        if (c.kt <= 32) {
            if (c.ka <= 32) {
                return std::make_unique<NativeBandPair<std::uint32_t, std::uint32_t>>(
                    c, s, p, in, out, extra);
            }
            return std::make_unique<NativeBandPair<std::uint32_t, std::uint64_t>>(
                c, s, p, in, out, extra);
        }
        if (c.ka <= 32) {
            return std::make_unique<NativeBandPair<std::uint64_t, std::uint32_t>>(
                c, s, p, in, out, extra);
        }
        return std::make_unique<NativeBandPair<std::uint64_t, std::uint64_t>>(
            c, s, p, in, out, extra);
    }
}
