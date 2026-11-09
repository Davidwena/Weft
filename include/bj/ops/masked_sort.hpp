#pragma once
#include "bj/common/timer.hpp"
#include "bj/ops/shared_sort.hpp"
#include "bj/core/multi_round.hpp"
#include "bj/ops/partition.hpp"
#include "bj/ops/quicksort.hpp"
#include "bj/proto/remask.hpp"
#include "bj/proto/shuffle.hpp"

namespace bj {
    inline std::size_t named_shuffle(const MaterialPlan &p, const std::string &name) {
        for (std::size_t i = 0; i < p.shuffles.size(); ++i) {
            if (p.shuffles[i].name == name) {
                return i;
            }
        }
        throw std::invalid_argument("unknown shuffle " + name);
    }

    inline std::vector<unsigned> table_widths_of(const MaskedTable &t) {
        std::vector<unsigned> w;
        for (const auto &c : t.columns) {
            w.push_back(c.width);
        }
        return w;
    }

    template <class T> class MaskedSortGroup {
        std::vector<MaskedTable *> inputs_;
        std::vector<std::unique_ptr<MaskedTable>> outputs_;
        unsigned party_, width_;
        std::vector<std::unique_ptr<RemaskStep>> remasks_;
        std::vector<Step *> steps_;
        std::unique_ptr<RoundExecutor> remask_round_;
        std::vector<std::unique_ptr<Shuffle>> shuffles_;
        std::vector<Shuffle *> shuffle_ptr_;
        std::unique_ptr<ShuffleGroup> shuffle_group_;
        std::vector<std::unique_ptr<PublicQuickSort<T>>> sorts_;
        std::vector<PublicQuickSort<T> *> sort_ptr_;
        std::vector<std::unique_ptr<FreshMsbPool<T>>> pools_;
        std::vector<FreshMsbPool<T> *> pool_ptr_;
        std::unique_ptr<SortGroup<T>> group_;
        std::vector<std::vector<unsigned>> columns_;
        bool used_ = false;
        SharedSortHandle *shared_ = nullptr;

    public:
        std::vector<SortStats> sort_stats;
        double shuffle_seconds = 0, sort_seconds = 0;

        MaskedSortGroup(std::span<MaskedTable *const> tables,
                        MaterialStore &s,
                        const MaterialPlan &p,
                        std::span<const std::string> names,
                        unsigned width)
            : inputs_(tables.begin(), tables.end()), party_(s.party()), width_(width) {
            if (tables.empty() || tables.size() > 2 || tables.size() != names.size()) {
                throw std::invalid_argument("masked sort group shape");
            }
            const auto N = tables[0]->columns[0].count;
            if (tables.size() == 1 && width == 64 && sort_binder()) {
                outputs_.push_back(std::make_unique<MaskedTable>(N, table_widths_of(*tables[0])));
                shared_ = (*sort_binder())(names[0], *tables[0], *outputs_[0]);
                if (shared_) {
                    sort_stats.emplace_back();
                    return;
                }
                outputs_.clear();
            }
            for (std::size_t r = 0; r < tables.size(); ++r) {
                auto &table = *tables[r];
                if (table.columns[0].count != N) {
                    throw std::invalid_argument("sort sizes differ");
                }
                outputs_.push_back(std::make_unique<MaskedTable>(N, table_widths_of(table)));
                std::vector<std::size_t> masks;
                for (std::size_t j = 0; j < table.columns.size(); ++j) {
                    masks.push_back(named_mask(p, names[r] + ".mask." + std::to_string(j)));
                }
                remasks_.push_back(std::make_unique<RemaskStep>(table, s, p, masks));
                steps_.push_back(remasks_.back().get());
                shuffles_.push_back(
                    std::make_unique<Shuffle>(table, s, p, named_shuffle(p, names[r])));
                shuffle_ptr_.push_back(shuffles_.back().get());
                sorts_.push_back(std::make_unique<PublicQuickSort<T>>(N));
                sort_ptr_.push_back(sorts_.back().get());
                pools_.push_back(std::make_unique<FreshMsbPool<T>>(
                    s, p, named_comparison(p, names[r] + ".compare"), N));
                pool_ptr_.push_back(pools_.back().get());
                columns_.emplace_back();
                for (unsigned j = 0; j < table.columns.size(); ++j) {
                    columns_.back().push_back(j);
                }
                sort_stats.emplace_back();
            }
            remask_round_ = std::make_unique<RoundExecutor>(steps_);
            shuffle_group_ = std::make_unique<ShuffleGroup>(shuffle_ptr_);
            group_ = std::make_unique<SortGroup<T>>(sort_ptr_, pool_ptr_, N, width);
        }

        void run(Net &net) {
            if (used_) {
                throw std::logic_error("masked sort already used");
            }
            used_ = true;
            if (shared_) {
                auto *hook = Net::thread_hook();
                if (!hook) {
                    throw std::logic_error("shared time sort requires multi scheduler");
                }
                hook->wait_shared(shared_->job);
                sort_stats[0] = shared_->stats;
                sort_seconds = shared_->seconds;
                return;
            }
            remask_round_->run(net);
            Timer timer;
            shuffle_group_->run(net);
            shuffle_seconds = timer.seconds();
            timer.reset();
            for (std::size_t r = 0; r < inputs_.size(); ++r) {
                const auto &col = inputs_[r]->columns[0];
                auto keys = sorts_[r]->keys();
                for (std::size_t i = 0; i < keys.size(); ++i) {
                    const auto d = col.width <= 32   ? uint128_t(col.d32.data()[i])
                                   : col.width <= 64 ? uint128_t(col.d64.data()[i])
                                                     : col.d128.data()[i];
                    const auto m = col.width <= 32   ? uint128_t(col.m32.data()[i])
                                   : col.width <= 64 ? uint128_t(col.m64.data()[i])
                                                     : col.m128.data()[i];
                    keys[i] = static_cast<T>((party_ == 0 ? d : uint128_t{0}) - m) &
                              native_mask<T>(width_);
                }
            }
            group_->run(net);
            for (std::size_t r = 0; r < inputs_.size(); ++r) {
                gather_columns(*inputs_[r], *outputs_[r], columns_[r], sorts_[r]->order());
                sort_stats[r] = sorts_[r]->stats(pools_[r]->capacity());
            }
            sort_seconds = timer.seconds();
        }

        MaskedTable &output(std::size_t r = 0) {
            return *outputs_.at(r);
        }
    };
}
