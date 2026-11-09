#pragma once
#include "bj/core/round_exec.hpp"
#include "bj/core/table.hpp"

namespace bj {

    void gather_columns(const MaskedTable &,
                        MaskedTable &,
                        std::span<const unsigned> columns,
                        std::span<const std::size_t> order);

    class RevealDestination : public Step {
        MaskedColumn &column_;
        unsigned party_;
        AlignedBuffer<std::uint32_t> share_, opened_;
        RingSlot slot_;
        bool used_ = false, done_ = false, bound_ = false;

    public:
        RevealDestination(MaskedColumn &c, unsigned party)
            : column_(c), party_(party), share_(c.count), opened_(c.count) {
            if (c.width != 32) {
                throw std::invalid_argument("destination ring must be32");
            }
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("destination bound twice");
            }
            bound_ = true;
            slot_ = p.ring(32, column_.count);
        }

        void contribute(OpenWriter &w) override {
            if (!bound_ || used_) {
                throw std::logic_error("destination material reused");
            }
            used_ = true;
            for (std::size_t i = 0; i < column_.count; ++i) {
                share_.data()[i] =
                    (party_ == 0 ? column_.d32.data()[i] : 0u) - column_.m32.data()[i];
            }
            w.ring<std::uint32_t>(slot_, share_.span());
        }

        void consume(OpenReader &r) override {
            if (!used_ || done_) {
                throw std::logic_error("destination consume state");
            }
            done_ = true;
            r.ring<std::uint32_t>(slot_, opened_.span());
        }

        std::span<const std::uint32_t> positions() const {
            return opened_.span();
        }
    };
}
