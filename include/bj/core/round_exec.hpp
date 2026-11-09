#pragma once
#include "bj/common/aligned.hpp"
#include "bj/common/packed.hpp"
#include "bj/net/net.hpp"
#include <optional>

namespace bj {
    struct BitSlot {
        std::size_t offset = 0, count = 0;
    };

    struct RingSlot {
        std::size_t field = 0, count = 0;
        unsigned width = 0;
    };

    class RoundPlan {
        OpenLayout layout_;
        std::size_t bits_ = 0, bit_field_ = 0;
        bool sealed_ = false;

    public:
        BitSlot bits(std::size_t count) {
            if (sealed_ || count > SIZE_MAX - bits_) {
                throw std::logic_error("round bit layout overflow/sealed");
            }
            BitSlot s{bits_, count};
            bits_ += count;
            return s;
        }

        RingSlot ring(unsigned width, std::size_t count) {
            if (sealed_) {
                throw std::logic_error("round layout sealed");
            }
            RingSlot s{layout_.fields().size(), count, width};
            layout_.add_ring(width, count);
            return s;
        }

        void seal() {
            if (sealed_) {
                return;
            }
            bit_field_ = layout_.fields().size();
            layout_.add_bits(bits_);
            layout_.seal();
            sealed_ = true;
        }

        const OpenLayout &layout() const noexcept {
            return layout_;
        }

        std::size_t bit_byte_offset() const noexcept {
            return layout_.fields()[bit_field_].offset;
        }

        std::size_t bit_count() const noexcept {
            return bits_;
        }
    };

    class OpenWriter {
        const RoundPlan &plan_;
        std::span<std::byte> data_;

    public:
        OpenWriter(const RoundPlan &plan, std::span<std::byte> data) : plan_(plan), data_(data) {
        }

        void bits(BitSlot slot, std::span<const std::uint64_t> value) {
            if (slot.offset > plan_.bit_count() || slot.count > plan_.bit_count() - slot.offset) {
                throw std::invalid_argument("bit slot outside round");
            }
            store_bits(data_.subspan(plan_.bit_byte_offset()), slot.offset, slot.count, value);
        }

        template <class T> void ring(RingSlot slot, std::span<const T> value) {
            const auto &f = plan_.layout().fields()[slot.field];
            if (f.width != slot.width || f.count != value.size()) {
                throw std::invalid_argument("ring slot shape");
            }
            pack_native<T>(slot.width, value, data_.subspan(f.offset, f.bytes));
        }
    };

    class OpenReader {
        const RoundPlan &plan_;
        std::span<const std::byte> data_;

    public:
        OpenReader(const RoundPlan &plan, std::span<const std::byte> data)
            : plan_(plan), data_(data) {
        }

        void bits(BitSlot slot, std::span<std::uint64_t> value) {
            if (slot.offset > plan_.bit_count() || slot.count > plan_.bit_count() - slot.offset) {
                throw std::invalid_argument("bit slot outside round");
            }
            load_bits(data_.subspan(plan_.bit_byte_offset()), slot.offset, slot.count, value);
        }

        template <class T> void ring(RingSlot slot, std::span<T> value) {
            const auto &f = plan_.layout().fields()[slot.field];
            if (f.width != slot.width || f.count != value.size()) {
                throw std::invalid_argument("ring slot shape");
            }
            unpack_native<T>(slot.width, data_.subspan(f.offset, f.bytes), value);
        }
    };

    struct Step {
        virtual ~Step() = default;
        virtual void bind(RoundPlan &) = 0;
        virtual void contribute(OpenWriter &) = 0;
        virtual void consume(OpenReader &) = 0;
    };

    class RoundExecutor {
        std::span<Step *const> steps_;
        RoundPlan plan_;
        AlignedBuffer<std::byte> share_, opened_;
        std::optional<OpenBuf> buf_;
        bool used_ = false, consumed_ = false;

    public:
        explicit RoundExecutor(std::span<Step *const> steps) : steps_(steps) {
            for (std::size_t i = 0; i < steps.size(); ++i) {
                if (!steps[i]) {
                    throw std::invalid_argument("null protocol batch");
                }
                for (std::size_t j = 0; j < i; ++j) {
                    if (steps[i] == steps[j]) {
                        throw std::invalid_argument("duplicate protocol batch");
                    }
                }
                steps[i]->bind(plan_);
            }
            plan_.seal();
            share_ = AlignedBuffer<std::byte>(plan_.layout().bytes());
            opened_ = AlignedBuffer<std::byte>(plan_.layout().bytes());
            std::fill(share_.span().begin(), share_.span().end(), std::byte{0});
            std::fill(opened_.span().begin(), opened_.span().end(), std::byte{0});
            buf_.emplace(plan_.layout(), share_.span(), opened_.span());
        }

        const RoundPlan &plan() const noexcept {
            return plan_;
        }

        void run_without_open() {
            if (plan_.layout().bits() != 0 || used_) {
                throw std::logic_error("local stage must be unused and contain zero open bits");
            }
            used_ = true;
            consumed_ = true;
            std::fill(share_.span().begin(), share_.span().end(), std::byte{0});
            OpenWriter writer(plan_, share_.span());
            for (auto *step : steps_) {
                step->contribute(writer);
            }
            OpenReader reader(plan_, share_.span());
            for (auto *step : steps_) {
                step->consume(reader);
            }
        }

        OpenBuf &prepare_open() {
            if (used_) {
                throw std::logic_error("one-time round already executed");
            }
            used_ = true;
            std::fill(share_.span().begin(), share_.span().end(), std::byte{0});
            OpenWriter writer(plan_, share_.span());
            for (auto *step : steps_) {
                step->contribute(writer);
            }
            return *buf_;
        }

        void consume_open() {
            if (!used_ || consumed_) {
                throw std::logic_error("round consume state");
            }
            consumed_ = true;
            OpenReader reader(plan_, opened_.span());
            for (auto *step : steps_) {
                step->consume(reader);
            }
        }

        void run(Net &net) {
            auto &buf = prepare_open();
            net.open_round(buf);
            consume_open();
        }
    };
}
