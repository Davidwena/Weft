#pragma once
#include "bj/core/table.hpp"
#include "bj/core/round_exec.hpp"

namespace bj {

    class RemaskStep : public Step {
        MaskedTable &table_;
        unsigned party_;
        std::vector<MaskedColumn> fresh_;
        std::vector<unsigned> columns_;
        std::vector<RingSlot> rings_;
        std::vector<BitSlot> bits_;
        bool bound_ = false, used_ = false, done_ = false;

    public:
        RemaskStep(MaskedTable &table,
                   MaterialStore &s,
                   const MaterialPlan &p,
                   std::span<const std::size_t> masks,
                   std::span<const unsigned> columns = {})
            : table_(table), party_(s.party()), rings_(masks.size()), bits_(masks.size()) {
            if (columns.empty()) {
                if (table.columns.size() != masks.size()) {
                    throw std::invalid_argument("remask shape");
                }
                for (unsigned j = 0; j < masks.size(); ++j) {
                    columns_.push_back(j);
                }
            } else {
                columns_.assign(columns.begin(), columns.end());
            }
            if (columns_.size() != masks.size()) {
                throw std::invalid_argument("remask selected columns");
            }
            fresh_.reserve(masks.size());
            for (std::size_t j = 0; j < masks.size(); ++j) {
                fresh_.emplace_back(table.columns.at(columns_[j]).width,
                                    table.columns.at(columns_[j]).count);
                load_mask_share(fresh_.back(), s, p, masks[j]);
            }
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("remask bound twice");
            }
            bound_ = true;
            for (std::size_t j = 0; j < fresh_.size(); ++j) {
                if (fresh_[j].width) {
                    rings_[j] = p.ring(fresh_[j].width, fresh_[j].count);
                } else {
                    bits_[j] = p.bits(fresh_[j].count);
                }
            }
        }

        void contribute(OpenWriter &w) override {
            if (!bound_ || used_) {
                throw std::logic_error("remask mask reused");
            }
            used_ = true;
            for (std::size_t j = 0; j < fresh_.size(); ++j) {
                auto &target = fresh_[j];
                const auto &source = table_.columns[columns_[j]];
                if (!target.width) {
                    auto out = target.bits.words();
                    auto d = source.bits.words(), old = source.mask_bits.words();
                    auto mask = target.mask_bits.words();
#pragma omp parallel for if (out.size() >= 16384) schedule(static)
                    for (std::size_t i = 0; i < out.size(); ++i) {
                        out[i] = old[i] ^ mask[i] ^ (party_ == 0 ? d[i] : 0);
                    }
                    w.bits(bits_[j], out);
                } else {
                    auto prepare = [&]<class T>() {
                        auto out = target.delta<T>();
                        auto d = source.delta<T>(), old = source.mask<T>();
                        auto mask = target.mask<T>();
                        const auto reduce = native_mask<T>(target.width);
#pragma omp parallel for if (out.size() >= 16384) schedule(static)
                        for (std::size_t i = 0; i < out.size(); ++i)
                            out[i] = ((party_ == 0 ? d[i] : T{0}) - old[i] + mask[i]) & reduce;
                        w.ring<T>(rings_[j], out);
                    };
                    if (target.width <= 32) {
                        prepare.template operator()<std::uint32_t>();
                    } else if (target.width <= 64) {
                        prepare.template operator()<std::uint64_t>();
                    } else {
                        prepare.template operator()<uint128_t>();
                    }
                }
            }
        }

        void consume(OpenReader &r) override {
            if (!used_ || done_) {
                throw std::logic_error("remask consume state");
            }
            done_ = true;
            for (std::size_t j = 0; j < fresh_.size(); ++j) {
                auto &dst = table_.columns[columns_[j]];
                const auto &mask = fresh_[j];
                if (!dst.width) {
                    r.bits(bits_[j], dst.bits.words());
                    std::copy(mask.mask_bits.words().begin(),
                              mask.mask_bits.words().end(),
                              dst.mask_bits.words().begin());
                } else {
                    auto finish = [&]<class T>() {
                        r.ring<T>(rings_[j], dst.delta<T>());
                        std::copy(
                            mask.mask<T>().begin(), mask.mask<T>().end(), dst.mask<T>().begin());
                    };
                    if (dst.width <= 32) {
                        finish.template operator()<std::uint32_t>();
                    } else if (dst.width <= 64) {
                        finish.template operator()<std::uint64_t>();
                    } else {
                        finish.template operator()<uint128_t>();
                    }
                }
                dst.dealer_knows_mask = true;
            }
            table_.dealer_knows_positions =
                std::all_of(table_.columns.begin(), table_.columns.end(), [](const auto &c) {
                    return c.dealer_knows_mask;
                });
        }
    };
}
