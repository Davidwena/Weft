#pragma once
#include "bj/proto/msb.hpp"
#include "bj/proto/shared_interval.hpp"
#include "bj/plan/investigation.hpp"

namespace bj {
    template <class T> class Interval : public Step {
        SharedIntervalHandle *shared_ = nullptr;
        unsigned k_, party_;
        std::size_t n_;
        std::span<const T> input_;
        std::vector<std::uint64_t> offsets_;
        std::vector<BandWindow> windows_;
        std::vector<AlignedBuffer<T>> shifted_;
        std::vector<BitVec> compared_;
        std::vector<std::unique_ptr<TableCompareStep<T>>> steps_;
        std::vector<std::span<std::uint64_t>> outputs_;
        bool bound_ = false, used_ = false, done_ = false;

    public:
        Interval(MaterialStore &store,
                 const MaterialPlan &p,
                 std::size_t index,
                 std::span<const T> masked,
                 std::span<const BandWindow> windows,
                 std::span<const std::span<std::uint64_t>> outputs)
            : k_(p.comparisons.at(index).width), party_(store.party()),
              n_(p.comparisons[index].count), input_(masked),
              windows_(windows.begin(), windows.end()), outputs_(outputs.begin(), outputs.end()) {
            const auto &shape = p.comparisons[index];
            if ((shape.kind != CompareKind::Interval && shape.kind != CompareKind::IntervalFresh) ||
                masked.size() != n_ || windows.size() != outputs.size()) {
                throw std::invalid_argument("interval shape");
            }
            for (auto window : windows) {
                if (window.lower > window.upper || window.upper == UINT64_MAX) {
                    throw std::invalid_argument("interval endpoints");
                }
            }
            for (auto out : outputs) {
                if (out.size() != bits_words(n_)) {
                    throw std::invalid_argument("interval output array");
                }
            }
            if (interval_binder()) {
                shared_ = (*interval_binder())(index, sizeof(T), masked.data(), windows, outputs);
                if (shared_) {
                    return;
                }
            }
            const auto &spec = p.comparisons[index];
            if ((spec.kind != CompareKind::Interval && spec.kind != CompareKind::IntervalFresh) ||
                masked.size() != n_ || windows.size() != outputs.size()) {
                throw std::invalid_argument("interval shape");
            }
            for (auto w : windows) {
                if (w.lower > w.upper || w.upper == UINT64_MAX) {
                    throw std::invalid_argument("interval endpoints");
                }
                offsets_.push_back(w.lower);
                offsets_.push_back(w.upper + 1);
            }
            std::sort(offsets_.begin(), offsets_.end());
            offsets_.erase(std::unique(offsets_.begin(), offsets_.end()), offsets_.end());
            if (spec.evaluations != offsets_.size()) {
                throw std::invalid_argument("interval distinct offset count");
            }
            shifted_.reserve(offsets_.size());
            compared_.reserve(offsets_.size());
            auto table = load_block_table(store, p, index);
            for (unsigned e = 0; e < offsets_.size(); ++e) {
                store.claim_comparison(index, e);
                shifted_.emplace_back(n_);
                compared_.emplace_back(n_);
                steps_.push_back(std::make_unique<TableCompareStep<T>>(
                    table,
                    store.packed(store.layout().comparisons[index].evaluations[e][0]),
                    party_,
                    shifted_.back().span(),
                    compared_.back().words()));
            }
            for (auto out : outputs_) {
                if (out.size() != bits_words(n_)) {
                    throw std::invalid_argument("interval output array");
                }
            }
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("interval bound twice");
            }
            bound_ = true;
            if (shared_) {
                return;
            }
            for (auto &s : steps_) {
                s->bind(p);
            }
        }

        void contribute(OpenWriter &w) override {
            if (!bound_ || used_) {
                throw std::logic_error("interval rho reused");
            }
            used_ = true;
            if (shared_) {
                shared_->requested = true;
                return;
            }
            for (std::size_t j = 0; j < offsets_.size(); ++j) {
                auto out = shifted_[j].span();
                const auto offset = static_cast<T>(offsets_[j]), mask = native_mask<T>(k_);
#pragma omp parallel for if (n_ >= 16384) schedule(static)
                for (std::size_t i = 0; i < n_; ++i) {
                    out[i] = (input_[i] - offset) & mask;
                }
                steps_[j]->contribute(w);
            }
        }

        void consume(OpenReader &r) override {
            if (!used_ || done_) {
                throw std::logic_error("interval consume state");
            }
            done_ = true;
            if (shared_) {
                if (!shared_->done) {
                    throw std::logic_error("shared interval dependency incomplete");
                }
                return;
            }
            for (auto &s : steps_) {
                s->consume(r);
            }
            combine();
        }

        void evaluate_local() {
            if (shared_) {
                throw std::logic_error("shared interval timing requires the joint batch");
            }
            if (!done_) {
                throw std::logic_error("local evaluation requires a completed interval");
            }
            for (std::size_t j = 0; j < steps_.size(); ++j) {
                steps_[j]->evaluate_local(compared_[j].words());
            }
            combine();
        }

        void combine() {
            if (shared_) {
                if (!shared_->done) {
                    throw std::logic_error("shared interval dependency incomplete");
                }
                return;
            }
            for (std::size_t j = 0; j < windows_.size(); ++j) {
                const auto lo = std::lower_bound(
                                    offsets_.begin(), offsets_.end(), windows_[j].lower) -
                                offsets_.begin(),
                           hi = std::lower_bound(
                                    offsets_.begin(), offsets_.end(), windows_[j].upper + 1) -
                                offsets_.begin();
                auto out = outputs_[j];
#pragma omp parallel for if (n_ >= 16384) schedule(static)
                for (std::size_t word = 0; word < out.size(); ++word) {
                    std::uint64_t top = 0;
                    if (party_ == 0) {
                        for (unsigned lane = 0; lane < 64 && word * 64 + lane < n_; ++lane) {
                            top |= std::uint64_t(((shifted_[lo].data()[word * 64 + lane] ^
                                                   shifted_[hi].data()[word * 64 + lane]) >>
                                                  (k_ - 1)) &
                                                 1)
                                   << lane;
                        }
                    }
                    out[word] = compared_[lo].words()[word] ^ compared_[hi].words()[word] ^ top;
                }
            }
        }
    };
}
