#include "bj/core/masked_material.hpp"
#include <numeric>
#include <cstring>

namespace bj {
    namespace {
        std::uint64_t uniform(const ComparisonRandom &random, std::uint64_t bound) {

            const auto reject = (std::uint64_t{0} - bound) % bound;
            std::uint64_t x;
            do {
                random(std::as_writable_bytes(std::span(&x, 1)));
            } while (x < reject);
            return x % bound;
        }

        void shuffle_indices(std::span<std::uint32_t> p, const ComparisonRandom &random) {
            std::iota(p.begin(), p.end(), 0u);
            for (std::size_t i = p.size(); i > 1; --i) {
                std::swap(p[i - 1], p[uniform(random, i)]);
            }
        }
    }

    void permutation_from_seed(std::span<const std::byte> seed, std::span<std::uint32_t> out) {
        if (seed.size() != 16) {
            throw std::invalid_argument("shuffle seed size");
        }
        Prg::Key key{};
        std::memcpy(key.data(), seed.data(), 16);
        Prg prg(key);
        std::array<std::byte, 4096> buffer;
        std::size_t used = buffer.size();
        std::uint64_t offset = 0;
        shuffle_indices(out, [&](auto bytes) {
            for (auto &b : bytes) {
                if (used == buffer.size()) {
                    prg.fill(0x53485546464c45ULL, offset, buffer);
                    offset += buffer.size();
                    used = 0;
                }
                b = buffer[used++];
            }
        });
    }

    void make_mask_values(const MaterialPlan &p,
                          std::vector<AlignedBuffer<std::byte>> &values,
                          const ComparisonRandom &random) {
        if (values.size() != p.inputs.size()) {
            throw std::invalid_argument("dealer mask inputs");
        }
        values.reserve(values.size() + p.masks.size());
        for (const auto &spec : p.masks) {
            const unsigned k = spec.width ? spec.width : 1;
            AlignedBuffer<std::byte> out(bits_bytes(checked_bits(spec.count, k)));
            if (spec.fresh) {
                random(out.span());
            } else {
                std::fill(out.span().begin(), out.span().end(), std::byte{0});
                for (const auto &term : spec.terms) {
                    const auto source = values.at(term.source).span();
                    unsigned source_k;
                    if (term.source < p.inputs.size()) {
                        source_k = p.inputs[term.source].width;
                    } else {
                        source_k = p.masks[term.source - p.inputs.size()].width;
                    }
                    source_k = source_k ? source_k : 1;
                    for (std::size_t j = 0; j < term.length; ++j) {
                        const auto i = term.destination_offset + j,
                                   from = term.source_offset + (j / term.divisor) % term.modulus;
                        const auto x = load_wide(source, from * source_k, source_k),
                                   old = load_wide(out.span(), i * k, k);
                        const auto v = spec.width
                                           ? (old + static_cast<uint128_t>(term.coefficient) * x) &
                                                 native_mask<uint128_t>(k)
                                           : (old ^ x);
                        store_wide(out.span(), i * k, k, v);
                    }
                }
            }
            if (spec.count * k % 8) {
                out.data()[out.size() - 1] &= std::byte((1u << (spec.count * k % 8)) - 1);
            }
            values.push_back(std::move(out));
        }
    }

    void make_shuffle_values(const MaterialPlan &p,
                             const MaterialLayout &l,
                             MaterialMode mode,
                             std::span<const std::span<const std::byte>> masks,
                             const ComparisonEmitter &emit,
                             const ComparisonRandom &random) {
        for (std::size_t s = 0; s < p.shuffles.size(); ++s) {
            const auto &spec = p.shuffles[s];
            const auto &ids = l.shuffles[s];
            AlignedBuffer<std::uint32_t> perm(spec.count);
            std::vector<AlignedBuffer<std::byte>> previous;
            for (auto col : spec.inputs) {
                previous.emplace_back(masks[col].size());
                std::copy(masks[col].begin(), masks[col].end(), previous.back().span().begin());
            }
            for (unsigned party = 0; party < p.n; ++party) {
                if (mode == MaterialMode::Prg) {
                    std::array<std::byte, 16> seed{};
                    random(seed);
                    permutation_from_seed(seed, perm.span());
                    emit(ids.permutations[party], seed);
                } else {
                    shuffle_indices(perm.span(), random);
                    AlignedBuffer<std::byte> packed(4 * spec.count);
                    pack_native<std::uint32_t>(32, perm.span(), packed.span());
                    emit(ids.permutations[party], packed.span());
                }
                for (std::size_t c = 0; c < spec.inputs.size(); ++c) {
                    auto col = spec.inputs[c];
                    const auto width = col < p.inputs.size() ? p.inputs[col].width
                                                             : p.masks[col - p.inputs.size()].width;
                    const auto k = width ? width : 1;
                    AlignedBuffer<std::byte> next(previous[c].size()), correction(next.size());
                    std::fill(correction.span().begin(), correction.span().end(), std::byte{0});
                    if (party + 1 == p.n) {
                        std::copy(masks[spec.outputs[c]].begin(),
                                  masks[spec.outputs[c]].end(),
                                  next.span().begin());
                    } else {
                        random(next.span());
                    }
                    for (std::size_t i = 0; i < spec.count; ++i) {
                        const auto a = load_wide(next.span(), i * k, k),
                                   b = load_wide(
                                       previous[c].span(), std::size_t(perm.data()[i]) * k, k);
                        store_wide(correction.span(),
                                   i * k,
                                   k,
                                   width ? (a - b) & native_mask<uint128_t>(k) : a ^ b);
                    }
                    emit(ids.corrections[party][c], correction.span());
                    previous[c] = std::move(next);
                }
            }
        }
    }
}
