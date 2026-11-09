#include "parallel.hpp"
#include "bj/core/comparison_dealer.hpp"
#include <bit>
#include "bj/proto/eda_shape.hpp"

namespace bj {
    CompareKind parse_compare_kind(const std::string &s) {
        if (s == "ltbits") {
            return CompareKind::LTBits;
        }
        if (s == "msb_known") {
            return CompareKind::MsbKnown;
        }
        if (s == "msb_fresh") {
            return CompareKind::MsbFresh;
        }
        if (s == "eq_fresh") {
            return CompareKind::EqFresh;
        }
        if (s == "eq") {
            return CompareKind::Eq;
        }
        if (s == "lts_known") {
            return CompareKind::LtsKnown;
        }
        if (s == "lts_fresh") {
            return CompareKind::LtsFresh;
        }
        if (s == "interval") {
            return CompareKind::Interval;
        }
        if (s == "interval_fresh") {
            return CompareKind::IntervalFresh;
        }
        if (s == "eda_msb") {
            return CompareKind::EdaMsb;
        }
        if (s == "eda_eq") {
            return CompareKind::EdaEq;
        }
        throw std::invalid_argument("unknown comparison protocol");
    }

    const char *compare_kind_name(CompareKind k) noexcept {
        switch (k) {
        case CompareKind::EqFresh:
            return "eq_fresh";
        case CompareKind::LTBits:
            return "ltbits";
        case CompareKind::MsbKnown:
            return "msb_known";
        case CompareKind::MsbFresh:
            return "msb_fresh";
        case CompareKind::Eq:
            return "eq";
        case CompareKind::LtsKnown:
            return "lts_known";
        case CompareKind::LtsFresh:
            return "lts_fresh";
        case CompareKind::EdaMsb:
            return "eda_msb";
        case CompareKind::EdaEq:
            return "eda_eq";
        case CompareKind::Interval:
            return "interval";
        default:
            return "interval_fresh";
        }
    }

    unsigned CompareSpec::block_width() const {
        if (d) {
            return d;
        }
        const bool msb = kind == CompareKind::MsbKnown || kind == CompareKind::MsbFresh ||
                         kind == CompareKind::Interval || kind == CompareKind::IntervalFresh;
        const auto ell = msb ? width - 1 : width;
        return ell ? best_compare_d(ell, (kind == CompareKind::Eq || kind == CompareKind::EqFresh))
                   : 1;
    }

    void make_comparison_evaluation(const CompareShape &s,
                                    std::size_t n,
                                    std::span<std::byte> out,
                                    const ComparisonRandom &random) {
        if (out.size() != bits_bytes(s.eval_bits() * n)) {
            throw std::invalid_argument("evaluation material size");
        }
        std::fill(out.begin(), out.end(), std::byte{0});
        auto masks = out.first(bits_bytes(s.inputs * n));
        random(masks);
        if (s.inputs * n % 8) {
            masks.back() &= std::byte((1u << (s.inputs * n % 8)) - 1);
        }
        for (std::size_t p = 0; p < s.products.size(); ++p) {
            for (std::size_t start = 0; start < n; start += 64) {
                const auto take = static_cast<unsigned>(std::min<std::size_t>(64, n - start));
                std::uint64_t word = width_mask(take);
                auto subset = s.products[p];
                while (subset) {
                    const auto j = std::countr_zero(subset);
                    word &= load_packed(out, j * n + start, take);
                    subset &= subset - 1;
                }
                store_packed(out, (s.inputs + p) * n + start, take, word);
            }
        }
    }

    void generate_comparison_material(const MaterialPlan &plan,
                                      const MaterialLayout &layout,
                                      std::span<const std::span<const std::byte>> masks,
                                      const ComparisonEmitter &emit,
                                      const ComparisonRandom &random) {
        for (std::size_t index = 0; index < plan.comparisons.size(); ++index) {
            const auto &c = plan.comparisons[index];
            const auto &ids = layout.comparisons[index];
            if (!c.evaluation_sources.empty()) {
                continue;
            }
            if (c.kind == CompareKind::EdaMsb || c.kind == CompareKind::EdaEq) {
                EdaShape shape(c.width, c.kind == CompareKind::EdaEq);
                AlignedBuffer<std::byte> packed(bits_bytes(c.width * c.count)),
                    bits(bits_bytes(c.width * c.count)),
                    triples(bits_bytes(3 * shape.gates * c.count));
                random(packed.span());
                emit(ids.aux[0], packed.span());
                std::fill(bits.span().begin(), bits.span().end(), std::byte{0});
                for (std::size_t i = 0; i < c.count; ++i) {
                    auto r = load_wide(packed.span(), i * c.width, c.width);
                    if (c.kind == CompareKind::EdaMsb) {
                        r = (uint128_t{0} - r) & native_mask<uint128_t>(c.width);
                    }
                    for (unsigned b = 0; b < c.width; ++b) {
                        store_packed(
                            bits.span(), b * c.count + i, 1, static_cast<std::uint64_t>(r >> b));
                    }
                }
                emit(ids.aux[1], bits.span());
                std::fill(triples.span().begin(), triples.span().end(), std::byte{0});
                AlignedBuffer<std::byte> a(bits_bytes(c.count)), b(bits_bytes(c.count));
                for (std::size_t g = 0; g < shape.gates; ++g) {
                    random(a.span());
                    random(b.span());
                    for (std::size_t i = 0; i < c.count; i += 64) {
                        auto take = static_cast<unsigned>(std::min<std::size_t>(64, c.count - i));
                        auto av = load_packed(a.span(), i, take),
                             bv = load_packed(b.span(), i, take);
                        store_packed(triples.span(), (3 * g) * c.count + i, take, av);
                        store_packed(triples.span(), (3 * g + 1) * c.count + i, take, bv);
                        store_packed(triples.span(), (3 * g + 2) * c.count + i, take, av & bv);
                    }
                }
                emit(ids.evaluations[0][0], triples.span());
                continue;
            }
            const bool msb = c.kind == CompareKind::MsbKnown || c.kind == CompareKind::MsbFresh ||
                             c.kind == CompareKind::Interval ||
                             c.kind == CompareKind::IntervalFresh;
            const bool lts = c.kind == CompareKind::LtsKnown || c.kind == CompareKind::LtsFresh;
            const bool fresh = c.kind == CompareKind::MsbFresh || c.kind == CompareKind::LtsFresh ||
                               c.kind == CompareKind::IntervalFresh ||
                               c.kind == CompareKind::EqFresh;
            const bool eq = c.kind == CompareKind::Eq || c.kind == CompareKind::EqFresh;
            const auto ell = msb ? c.width - 1 : c.width;
            auto generate = [&]<class T>() {
                AlignedBuffer<T> r(c.count), rp(lts ? c.count : 0), threshold(c.count);
                AlignedBuffer<std::byte> packed(bits_bytes(c.count * c.width)),
                    sign(bits_bytes(c.count));
                const auto mask = native_mask<T>(c.width);
                if (fresh) {
                    random(packed.span());
                    x11_unpack<T>(c.width, packed.span(), r.span());
                } else {
                    x11_unpack<T>(c.width, masks[c.inputs[lts ? 1 : 0]], r.span());
                }
                if (lts) {
                    if (fresh) {
                        random(packed.span());
                        x11_unpack<T>(c.width, packed.span(), rp.span());
                    } else {
                        x11_unpack<T>(c.width, masks[c.inputs[0]], rp.span());
                        for (auto &x : rp.span()) {
                            x = (T{0} - x) & mask;
                        }
                    }
                }
                std::size_t aux = 0;
                if (fresh) {
                    x11_pack<T>(c.width, r.span(), packed.span());
                    emit(ids.aux[aux++], packed.span());
                    if (lts) {
                        x11_pack<T>(c.width, rp.span(), packed.span());
                        emit(ids.aux[aux++], packed.span());
                    }
                }
                std::fill(sign.span().begin(), sign.span().end(), std::byte{0});
                if (msb) {
#pragma omp parallel for if (x11_parallel(c.count)) schedule(static)
                    for (std::size_t w = 0; w < sign.size(); ++w) {
                        unsigned value = 0;
                        for (unsigned lane = 0; lane < 8 && 8 * w + lane < c.count; ++lane) {
                            auto i = 8 * w + lane;
                            const T negative = (T{0} - r.data()[i]) & mask;
                            value |= unsigned(negative >> (c.width - 1)) << lane;
                            threshold.data()[i] = c.width > 1
                                                      ? native_mask<T>(c.width - 1) -
                                                            (negative & native_mask<T>(c.width - 1))
                                                      : 0;
                        }
                        sign.data()[w] = std::byte(value);
                    }
                }
                if (lts)
                    for (std::size_t i = 0; i < c.count; ++i) {
                        const T sum = (r.data()[i] + rp.data()[i]) & mask;
                        const bool carry = sum < r.data()[i];
                        store_packed(sign.span(), i, 1, carry);
                    }
                if ((msb && c.kind != CompareKind::Interval &&
                     c.kind != CompareKind::IntervalFresh) ||
                    lts)
                    emit(ids.aux[aux], sign.span());
                if (!ell)
                    return;
                CompareShape shape(ell, c.block_width(), eq);
                AlignedBuffer<std::byte> table(bits_bytes(shape.table_bits() * c.count));
                if (msb) {
                    if (c.table_source.empty()) {
                        x11_table<T>(shape, threshold.span(), table.span());
                        emit(ids.tables[0], table.span());
                    }
                } else {
                    x11_table<T>(shape, r.span(), table.span());
                    emit(ids.tables[0], table.span());
                }
                if (lts) {
                    x11_table<T>(shape, rp.span(), table.span());
                    emit(ids.tables[1], table.span());
                    for (std::size_t i = 0; i < c.count; ++i)
                        threshold.data()[i] = (r.data()[i] + rp.data()[i]) & mask;
                    x11_table<T>(shape, threshold.span(), table.span());
                    emit(ids.tables[2], table.span());
                }
                AlignedBuffer<std::byte> evaluation(bits_bytes(shape.eval_bits() * c.count));
                for (const auto &round : ids.evaluations)
                    for (auto segment : round) {
                        x11_evaluation(shape, c.count, evaluation.span(), random);
                        emit(segment, evaluation.span());
                    }
            };
            if (c.width <= 32) {
                generate.template operator()<std::uint32_t>();
            } else if (c.width <= 64) {
                generate.template operator()<std::uint64_t>();
            } else {
                generate.template operator()<uint128_t>();
            }
        }
    }
}
