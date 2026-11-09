#pragma once
#include "bj/core/round_exec.hpp"
#include "bj/core/material.hpp"
#include "bj/proto/compare_shape.hpp"
#include <array>
#include <bit>
#include <memory>

namespace bj {
    inline const MaterialPlan &
    require_comparison(const MaterialPlan &p, std::size_t c, CompareKind kind, unsigned width) {
        const auto &s = p.comparisons.at(c);
        if (s.kind != kind || s.width != width) {
            throw std::invalid_argument("typed comparison kind/width mismatch");
        }
        return p;
    }

    inline std::uint64_t low_mask64(unsigned n) noexcept {
        return (UINT64_MAX >> ((64 - n) & 63)) & (std::uint64_t{0} - std::uint64_t(n != 0));
    }

    class BlockTable {
        CompareShape shape_;
        std::size_t n_, word_count_;
        AlignedBuffer<std::uint64_t> entries_;
        AlignedBuffer<std::uint8_t> prefix_;

    public:
        BlockTable(const CompareShape &shape, std::size_t n, std::span<const std::byte> packed)
            : shape_(shape), n_(n), word_count_(((std::size_t{1} << shape.d) + 63) >> 6),
              entries_(shape.blocks * word_count_ * n),
              prefix_(shape.blocks * (word_count_ + 1) * n) {
            if (packed.size() != bits_bytes(shape.table_bits() * n)) {
                throw std::invalid_argument("block table share size");
            }
            const unsigned entry_bits = 1u << shape.d;
#pragma omp parallel for if (n >= 16384) schedule(static)
            for (std::size_t i = 0; i < n; ++i) {
                for (unsigned b = 0; b < shape.blocks; ++b) {
                    std::uint8_t parity = 0;
                    prefix_.data()[b * (word_count_ + 1) * n + i] = 0;
                    for (std::size_t w = 0; w < word_count_; ++w) {
                        const auto value = load_packed(
                            packed, (b * n + i) * entry_bits + w * 64, std::min(64u, entry_bits));
                        entries_.data()[(b * word_count_ + w) * n + i] = value;
                        parity ^= static_cast<std::uint8_t>(std::popcount(value) & 1);
                        prefix_.data()[(b * (word_count_ + 1) + w + 1) * n + i] = parity;
                    }
                }
            }
        }

        const CompareShape &shape() const noexcept {
            return shape_;
        }

        std::size_t size() const noexcept {
            return n_;
        }

        std::pair<unsigned, unsigned>
        lookup(unsigned b, std::size_t i, uint128_t R, bool reverse) const noexcept {
            const auto digit = (R >> ((shape_.blocks - 1 - b) * shape_.d)) & ((1u << shape_.d) - 1);
            const auto w = digit >> 6;
            const auto bit = static_cast<unsigned>(digit & 63);
            const auto word = entries_.data()[(b * word_count_ + w) * n_ + i];
            const unsigned eq = static_cast<unsigned>((word >> bit) & 1);
            const auto before = prefix_.data()[(b * (word_count_ + 1) + w) * n_ + i];
            const auto lt = reverse
                                ? (prefix_.data()[(b * (word_count_ + 1) + word_count_) * n_ + i] ^
                                   before ^ (std::popcount(word & low_mask64(bit + 1)) & 1))
                                : (before ^ (std::popcount(word & low_mask64(bit)) & 1));
            return {eq, static_cast<unsigned>(lt)};
        }
    };

    class BooleanGateTables {
        CompareShape shape_;
        std::size_t n_, words_, f_words_;
        AlignedBuffer<std::uint64_t> rho_, f_;

    public:
        BooleanGateTables(const CompareShape &shape,
                          std::size_t n,
                          unsigned party,
                          std::span<const std::byte> packed)
            : shape_(shape), n_(n), words_(bits_words(n)), f_words_(bits_words(shape.f_bits)),
              rho_(shape.inputs * words_), f_(f_words_ * n) {
            if (packed.size() != bits_bytes(shape.eval_bits() * n)) {
                throw std::invalid_argument("Boolean evaluation material size");
            }
            for (unsigned u = 0; u < shape.inputs; ++u) {
                load_bits(packed, u * n, n, rho_.span().subspan(u * words_, words_));
            }
            std::fill(f_.span().begin(), f_.span().end(), 0);

#pragma omp parallel for if (n >= 16384) schedule(static)
            for (std::size_t block = 0; block < words_; ++block) {
                const auto base = block * 64;
                const auto take = static_cast<unsigned>(std::min<std::size_t>(64, n - base));
                std::array<std::uint64_t, 2048> coefficients;
                for (const auto &gate : shape_.gates) {
                    const auto count = gate.columns.size();
                    coefficients[0] = party == 0 ? low_mask64(take) : 0;
                    for (std::size_t s = 1; s < count; ++s) {
                        coefficients[s] = load_packed(packed, gate.columns[s] * n + base, take);
                    }
                    for (unsigned bit = 0; bit < gate.inputs.size(); ++bit) {
                        for (unsigned s = 0; s < count; ++s) {
                            if (!(s & (1u << bit))) {
                                coefficients[s] ^= coefficients[s | (1u << bit)];
                            }
                        }
                    }
                    for (std::size_t z = 0; z < count; ++z) {
                        const auto pos = gate.f_offset + z;
                        auto value = coefficients[z];
                        while (value) {
                            const auto lane = std::countr_zero(value);
                            f_.data()[(pos / 64) * n + base + lane] |= std::uint64_t{1}
                                                                       << (pos % 64);
                            value &= value - 1;
                        }
                    }
                }
            }
        }

        std::span<const std::uint64_t> rho(unsigned u) const noexcept {
            return rho_.span().subspan(u * words_, words_);
        }

        unsigned evaluate(std::size_t i, std::uint32_t hat) const noexcept {
            unsigned result = 0;
            for (unsigned b = 0; b < shape_.gates.size(); ++b) {
                const auto &gate = shape_.gates[b];
                const auto local =
                    shape_.equality
                        ? hat
                        : ((hat & ((1u << b) - 1)) | (((hat >> (shape_.blocks - 1 + b)) & 1) << b));
                const auto z = (~local) & ((1u << gate.inputs.size()) - 1);
                const auto pos = gate.f_offset + z;
                result ^= static_cast<unsigned>((f_.data()[(pos / 64) * n_ + i] >> (pos % 64)) & 1);
            }
            return result;
        }
    };

    template <class T> class TableCompareStep : public Step {
        std::shared_ptr<const BlockTable> table_;
        BooleanGateTables gates_;
        std::span<const T> R_;
        std::span<std::uint64_t> output_;
        std::size_t n_, words_;
        bool reverse_, bound_ = false, used_ = false, consumed_ = false;
        std::array<BitSlot, 21> slots_{};
        AlignedBuffer<std::uint64_t> masked_;

    public:
        TableCompareStep(std::shared_ptr<const BlockTable> table,
                         std::span<const std::byte> evaluation,
                         unsigned party,
                         std::span<const T> R,
                         std::span<std::uint64_t> output,
                         bool reverse = false)
            : table_(std::move(table)), gates_(table_->shape(), table_->size(), party, evaluation),
              R_(R), output_(output), n_(table_->size()), words_(bits_words(n_)), reverse_(reverse),
              masked_(table_->shape().inputs * words_) {
            if (table_->shape().ell > sizeof(T) * 8 || R.size() != n_ || output.size() != words_) {
                throw std::invalid_argument("comparison batch shape");
            }
        }

        void bind(RoundPlan &p) override {
            if (bound_) {
                throw std::logic_error("comparison step already bound");
            }
            bound_ = true;
            for (unsigned u = 0; u < table_->shape().inputs; ++u) {
                slots_[u] = p.bits(n_);
            }
        }

        void contribute(OpenWriter &writer) override {
            if (!bound_ || used_) {
                throw std::logic_error("comparison masks already used");
            }
            used_ = true;
            const auto &s = table_->shape();
#pragma omp parallel for if (n_ >= 16384) schedule(static)
            for (std::size_t w = 0; w < words_; ++w) {
                std::array<std::uint64_t, 21> bits{};
                for (unsigned lane = 0; lane < 64 && w * 64 + lane < n_; ++lane) {
                    const auto i = w * 64 + lane;
                    const auto R = R_[i] & native_mask<T>(s.ell);
                    for (unsigned b = 0; b < s.blocks; ++b) {
                        const auto [eq, lt] = table_->lookup(b, i, R, reverse_);
                        if (s.equality) {
                            bits[b] |= std::uint64_t(eq) << lane;
                        } else {
                            if (b + 1 < s.blocks) {
                                bits[b] |= std::uint64_t(eq) << lane;
                            }
                            bits[s.blocks - 1 + b] |= std::uint64_t(lt) << lane;
                        }
                    }
                }
                for (unsigned u = 0; u < s.inputs; ++u) {
                    masked_.data()[u * words_ + w] = bits[u] ^ gates_.rho(u)[w];
                }
            }
            for (unsigned u = 0; u < s.inputs; ++u) {
                writer.bits(slots_[u], masked_.span().subspan(u * words_, words_));
            }
        }

        void evaluate_local(std::span<std::uint64_t> out) const {
            if (out.size() != words_) {
                throw std::invalid_argument("comparison result size");
            }
            const auto ucount = table_->shape().inputs;
#ifndef BJ_PROFILE_SERIAL
#pragma omp parallel for if (n_ >= 16384) schedule(static)
#endif
            for (std::size_t w = 0; w < words_; ++w) {
                std::uint64_t value = 0;
                for (unsigned lane = 0; lane < 64 && w * 64 + lane < n_; ++lane) {
                    std::uint32_t hat = 0;
                    for (unsigned u = 0; u < ucount; ++u) {
                        hat |=
                            static_cast<std::uint32_t>((masked_.data()[u * words_ + w] >> lane) & 1)
                            << u;
                    }
                    value |= std::uint64_t(gates_.evaluate(w * 64 + lane, hat)) << lane;
                }
                out[w] = value;
            }
        }

        void consume(OpenReader &reader) override {
            if (!used_ || consumed_) {
                throw std::logic_error("comparison consume state");
            }
            consumed_ = true;
            for (unsigned u = 0; u < table_->shape().inputs; ++u) {
                reader.bits(slots_[u], masked_.span().subspan(u * words_, words_));
            }
            evaluate_local(output_);
        }
    };

    inline std::shared_ptr<const BlockTable> load_block_table(MaterialStore &store,
                                                              const MaterialPlan &p,
                                                              std::size_t c,
                                                              unsigned table = 0) {
        const auto &spec = p.comparisons.at(c);
        const bool msb = spec.kind == CompareKind::MsbKnown || spec.kind == CompareKind::MsbFresh ||
                         spec.kind == CompareKind::Interval ||
                         spec.kind == CompareKind::IntervalFresh;
        CompareShape shape(msb ? spec.width - 1 : spec.width,
                           spec.block_width(),
                           (spec.kind == CompareKind::Eq || spec.kind == CompareKind::EqFresh));
        return std::make_shared<BlockTable>(
            shape, spec.count, store.packed(store.layout().comparisons[c].tables.at(table)));
    }

    template <class T> class LTBitsBatch : public TableCompareStep<T> {
        using Base = TableCompareStep<T>;

        static std::span<const std::byte>
        claim(MaterialStore &store, std::size_t c, unsigned evaluation) {
            store.claim_comparison(c, evaluation);
            return store.packed(store.layout().comparisons.at(c).evaluations.at(evaluation).at(0));
        }

    public:
        LTBitsBatch(MaterialStore &store,
                    const MaterialPlan &p,
                    std::size_t c,
                    unsigned evaluation,
                    std::span<const T> R,
                    std::span<std::uint64_t> out,
                    bool reverse = false,
                    std::shared_ptr<const BlockTable> table = {})
            : Base(table ? table : load_block_table(store, p, c),
                   claim(store, c, evaluation),
                   store.party(),
                   R,
                   out,
                   reverse) {
            if (p.comparisons.at(c).kind != CompareKind::LTBits) {
                throw std::invalid_argument("expected LTBits material");
            }
        }
    };
}
