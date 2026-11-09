#pragma once
#include "bj/proto/msb.hpp"
#include <numeric>

namespace bj {

    class BoundedOpen {
        OpenLayout layout_;
        AlignedBuffer<std::byte> tx_, rx_;

    public:
        BoundedOpen(unsigned width, std::size_t max_count, unsigned fields) {
            for (unsigned i = 0; i < fields; ++i) {
                if (width) {
                    layout_.add_ring(width, max_count);
                } else {
                    layout_.add_bits(max_count);
                }
            }
            layout_.seal(true);
            tx_ = AlignedBuffer<std::byte>(layout_.bytes());
            rx_ = AlignedBuffer<std::byte>(layout_.bytes());
        }

        void shape(std::span<const std::size_t> counts) {
            layout_.resize_counts(counts);
            std::fill(tx_.span().begin(), tx_.span().end(), std::byte{0});
        }

        std::span<std::byte> send(unsigned field) {
            const auto &f = layout_.fields()[field];
            return tx_.span().subspan(f.offset, f.bytes);
        }

        std::span<const std::byte> received(unsigned field) const {
            const auto &f = layout_.fields()[field];
            return rx_.span().subspan(f.offset, f.bytes);
        }

        void run(Net &net) {
            OpenBuf b(
                layout_, tx_.span().first(layout_.bytes()), rx_.span().first(layout_.bytes()));
            net.open_round(b);
        }
    };

    class MaterialBudgetExhausted : public std::runtime_error {
    public:
        MaterialBudgetExhausted()
            : std::runtime_error(
                  "DC comparison pool exhausted; all banks output bottom, discard material") {
        }
    };

    struct SortStats {
        std::size_t comparisons = 0, capacity = 0;
        unsigned depth = 0;
    };

    template <class T> class FreshMsbPool {
        unsigned width_, party_;
        std::size_t capacity_, used_ = 0, start_ = 0, count_ = 0, max_batch_;
        unsigned phase_ = 0;
        std::shared_ptr<const BlockTable> table_;
        BooleanGateTables gates_;
        AlignedBuffer<T> r_, hat_;
        BitVec sign_, masked_, result_;

    public:
        FreshMsbPool(MaterialStore &s, const MaterialPlan &p, std::size_t c, std::size_t max_batch)
            : width_(p.comparisons.at(c).width), party_(s.party()),
              capacity_(p.comparisons[c].count), max_batch_(max_batch),
              table_(load_block_table(s, p, c)),
              gates_(table_->shape(),
                     capacity_,
                     party_,
                     s.packed(s.layout().comparisons[c].evaluations[0][0])),
              r_(capacity_), hat_(max_batch), sign_(capacity_),
              masked_(table_->shape().inputs * bits_words(max_batch) * 64), result_(max_batch) {
            if (p.comparisons[c].kind != CompareKind::MsbFresh ||
                sizeof(T) != (width_ <= 64 ? 8u : 16u)) {
                throw std::invalid_argument("fresh sort pool kind/width");
            }
            s.claim_comparison(c, 0);
            unpack_native<T>(width_, s.packed(s.layout().comparisons[c].aux[0]), r_.span());
            sign_.unpack(s.packed(s.layout().comparisons[c].aux[1]));
        }

        unsigned inputs() const {
            return table_->shape().inputs;
        }

        std::size_t used() const {
            return used_;
        }

        std::size_t capacity() const {
            return capacity_;
        }

        void reserve(std::size_t n) {
            if (phase_ || n > max_batch_) {
                throw std::logic_error("DC sort pool phase/batch");
            }

            if (n > capacity_ - used_) {
                throw MaterialBudgetExhausted();
            }
            start_ = used_;
            used_ += n;
            count_ = n;
            phase_ = 1;
        }

        void opening(std::span<const T> diff, std::span<std::byte> wire) {
            if (phase_ != 1 || diff.size() != count_) {
                throw std::logic_error("pool opening state");
            }
            for (std::size_t i = 0; i < count_; ++i) {
                hat_.data()[i] = (diff[i] + r_.data()[start_ + i]) & native_mask<T>(width_);
            }
            pack_native<T>(width_, hat_.span().first(count_), wire);
            phase_ = 2;
        }

        void table_opening(std::span<const std::byte> opened, std::span<std::byte> wire) {
            if (phase_ != 2) {
                throw std::logic_error("pool table state");
            }
            unpack_native<T>(width_, opened, hat_.span().first(count_));
            const auto &sh = table_->shape();
            const auto words = bits_words(count_);
            std::fill(masked_.words().begin(), masked_.words().end(), 0);
#pragma omp parallel for if (count_ >= 16384) schedule(static)
            for (std::size_t w = 0; w < words; ++w) {
                std::array<std::uint64_t, 21> values{};
                for (unsigned lane = 0; lane < 64 && w * 64 + lane < count_; ++lane) {
                    const auto i = w * 64 + lane, global = start_ + i;
                    const auto R = hat_.data()[i] & native_mask<T>(width_ - 1);
                    for (unsigned b = 0; b < sh.blocks; ++b) {
                        auto [eq, lt] = table_->lookup(b, global, R, false);
                        if (b + 1 < sh.blocks) {
                            values[b] |= std::uint64_t(eq) << lane;
                        }
                        values[sh.blocks - 1 + b] |= std::uint64_t(lt) << lane;
                    }
                    for (unsigned u = 0; u < sh.inputs; ++u) {
                        values[u] ^= ((gates_.rho(u)[global / 64] >> (global % 64)) & 1) << lane;
                    }
                }
                for (unsigned u = 0; u < sh.inputs; ++u) {
                    masked_.words()[u * bits_words(max_batch_) + w] = values[u];
                }
            }
            for (unsigned u = 0; u < sh.inputs; ++u) {
                store_bits(wire,
                           u * count_,
                           count_,
                           masked_.words().subspan(u * bits_words(max_batch_), words));
            }
            phase_ = 3;
        }

        void reveal(std::span<const std::byte> opened, std::span<std::byte> wire) {
            if (phase_ != 3) {
                throw std::logic_error("pool reveal state");
            }
            const auto &sh = table_->shape();
            const auto words = bits_words(count_);
            for (unsigned u = 0; u < sh.inputs; ++u) {
                load_bits(opened,
                          u * count_,
                          count_,
                          masked_.words().subspan(u * bits_words(max_batch_), words));
            }
#pragma omp parallel for if (count_ >= 16384) schedule(static)
            for (std::size_t w = 0; w < words; ++w) {
                std::uint64_t out = 0;
                for (unsigned lane = 0; lane < 64 && w * 64 + lane < count_; ++lane) {
                    const auto i = w * 64 + lane, global = start_ + i;
                    std::uint32_t hat = 0;
                    for (unsigned u = 0; u < sh.inputs; ++u) {
                        hat |= std::uint32_t(
                                   (masked_.words()[u * bits_words(max_batch_) + w] >> lane) & 1)
                               << u;
                    }
                    const auto bit =
                        gates_.evaluate(global, hat) ^ unsigned(sign_.get(global)) ^
                        (party_ == 0 ? unsigned((hat_.data()[i] >> (width_ - 1)) & 1) : 0u);
                    out |= std::uint64_t(bit) << lane;
                }
                result_.words()[w] = out;
            }
            store_bits(wire, 0, count_, result_.words().first(words));
            phase_ = 4;
        }

        void finish(std::span<const std::byte> opened) {
            if (phase_ != 4) {
                throw std::logic_error("pool finish state");
            }
            load_bits(opened, 0, count_, result_.words().first(bits_words(count_)));
            phase_ = 0;
        }

        bool less(std::size_t i) const {
            return result_.get(i);
        }
    };

    template <class T> class PublicQuickSort {
        struct Segment {
            std::size_t first, last;
        };

        AlignedBuffer<std::size_t> order_, scratch_;
        AlignedBuffer<Segment> segments_, next_;
        AlignedBuffer<T> keys_, diff_;
        std::size_t n_, segments_count_ = 0, comparisons_ = 0;
        unsigned depth_ = 0;

    public:
        explicit PublicQuickSort(std::size_t n)
            : order_(n), scratch_(n), segments_(n), next_(n), keys_(n), diff_(n), n_(n) {
            std::iota(order_.span().begin(), order_.span().end(), 0);
            if (n > 1) {
                segments_.data()[0] = {0, n};
                segments_count_ = 1;
            }
        }

        std::span<const T> differences(std::size_t count) const {
            return diff_.span().first(count);
        }

        std::span<T> keys() {
            return keys_.span();
        }

        std::span<const std::size_t> order() const {
            return order_.span();
        }

        bool active() const {
            return segments_count_ != 0;
        }

        std::span<const T> prepare() {
            std::size_t m = 0;
            for (std::size_t s = 0; s < segments_count_; ++s) {
                auto [a, b] = segments_.data()[s];
                const auto pivot = keys_.data()[order_.data()[a]];
                for (std::size_t j = a + 1; j < b; ++j) {
                    diff_.data()[m++] = keys_.data()[order_.data()[j]] - pivot;
                }
            }
            comparisons_ += m;
            if (m) {
                ++depth_;
            }
            return diff_.span().first(m);
        }

        void partition(const FreshMsbPool<T> &pool) {
            std::copy(order_.span().begin(), order_.span().end(), scratch_.span().begin());
            std::size_t consumed = 0, next_count = 0;
            for (std::size_t s = 0; s < segments_count_; ++s) {
                auto [a, b] = segments_.data()[s];
                const auto base = consumed;
                std::size_t left = 0;
                for (std::size_t j = a + 1; j < b; ++j) {
                    left += pool.less(consumed++);
                }
                const auto pivot = a + left;
                std::size_t lo = a, hi = pivot + 1;
                for (std::size_t j = a + 1; j < b; ++j) {
                    scratch_.data()[pool.less(base + j - a - 1) ? lo++ : hi++] = order_.data()[j];
                }
                scratch_.data()[pivot] = order_.data()[a];
                if (pivot - a > 1) {
                    next_.data()[next_count++] = {a, pivot};
                }
                if (b - pivot - 1 > 1) {
                    next_.data()[next_count++] = {pivot + 1, b};
                }
            }
            std::swap(order_, scratch_);
            std::swap(segments_, next_);
            segments_count_ = next_count;
        }

        SortStats stats(std::size_t capacity) const {
            return {comparisons_, capacity, depth_};
        }
    };

    template <class T> class SortGroup {
        std::vector<PublicQuickSort<T> *> sorts_;
        std::vector<FreshMsbPool<T> *> pools_;
        BoundedOpen first_, second_, third_;
        std::array<std::size_t, 2> count_{}, bits_{};

        static unsigned checked_inputs(std::span<FreshMsbPool<T> *const> pools) {
            if (pools.empty() || pools.size() > 2) {
                throw std::invalid_argument("sort pool group shape");
            }
            for (auto *pool : pools) {
                if (!pool) {
                    throw std::invalid_argument("null sort pool");
                }
            }
            return pools[0]->inputs();
        }

    public:
        SortGroup(std::span<PublicQuickSort<T> *const> sorts,
                  std::span<FreshMsbPool<T> *const> pools,
                  std::size_t n,
                  unsigned width)
            : sorts_(sorts.begin(), sorts.end()), pools_(pools.begin(), pools.end()),
              first_(width, n, static_cast<unsigned>(sorts.size())),
              second_(0, n * checked_inputs(pools), static_cast<unsigned>(sorts.size())),
              third_(0, n, static_cast<unsigned>(sorts.size())) {
            if (sorts.empty() || sorts.size() > 2 || sorts.size() != pools.size() ||
                std::any_of(sorts.begin(), sorts.end(), [](auto *sort) {
                    return sort == nullptr;
                })) {
                throw std::invalid_argument("sort group shape");
            }
        }

        void run(Net &net) {
            while (std::any_of(sorts_.begin(), sorts_.end(), [](auto *s) {
                return s->active();
            })) {
                for (unsigned j = 0; j < sorts_.size(); ++j) {
                    count_[j] = sorts_[j]->prepare().size();
                    bits_[j] = count_[j] * pools_[j]->inputs();
                    pools_[j]->reserve(count_[j]);
                }
                const auto counts = std::span(count_).first(sorts_.size());
                first_.shape(counts);
                second_.shape(std::span(bits_).first(sorts_.size()));
                third_.shape(counts);

                for (unsigned j = 0; j < sorts_.size(); ++j) {
                    pools_[j]->opening(sorts_[j]->differences(count_[j]), first_.send(j));
                }
                first_.run(net);
                for (unsigned j = 0; j < sorts_.size(); ++j) {
                    pools_[j]->table_opening(first_.received(j), second_.send(j));
                }
                second_.run(net);
                for (unsigned j = 0; j < sorts_.size(); ++j) {
                    pools_[j]->reveal(second_.received(j), third_.send(j));
                }
                third_.run(net);
                for (unsigned j = 0; j < sorts_.size(); ++j) {
                    pools_[j]->finish(third_.received(j));
                    sorts_[j]->partition(*pools_[j]);
                }
            }
        }
    };

    std::size_t dc_pool_capacity(std::size_t n, double factor, bool safe);
}
