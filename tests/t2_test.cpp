#include "bj/proto/msb.hpp"
#include "bj/proto/eq.hpp"
#include "bj/proto/lts.hpp"
#include "bj/proto/eda.hpp"
#include "bj/ref/plaintext.hpp"
#include <unistd.h>
#include "bj/core/comparison_dealer.hpp"
#include "allocation.hpp"
#include <gtest/gtest.h>
#include <omp.h>

namespace {
    using namespace bj;

    struct Random {
        Prg prg{Prg::Key{}};
        std::uint64_t domain = 0;

        void fill(std::span<std::byte> bytes) {
            prg.fill(++domain, 0, bytes);
        }
    };

    using Bytes = std::vector<std::byte>;

    std::vector<Bytes> split_bits(std::span<const std::byte> value, unsigned n, Random &random) {
        std::vector<Bytes> out(n, Bytes(value.size()));
        out.back().assign(value.begin(), value.end());
        for (unsigned p = 0; p + 1 < n; ++p) {
            random.fill(out[p]);
            for (std::size_t i = 0; i < value.size(); ++i) {
                out.back()[i] ^= out[p][i];
            }
        }
        return out;
    }

    template <class T>
    std::vector<std::vector<T>>
    split_ring(std::span<const T> value, unsigned k, unsigned n, Random &random) {
        std::vector<std::vector<T>> out(n, std::vector<T>(value.size()));
        out.back().assign(value.begin(), value.end());
        const auto mask = native_mask<T>(k);
        for (unsigned p = 0; p + 1 < n; ++p) {
            random.fill(std::as_writable_bytes(std::span(out[p])));
            for (std::size_t i = 0; i < value.size(); ++i) {
                out[p][i] &= mask;
                out.back()[i] = (out.back()[i] - out[p][i]) & mask;
            }
        }
        return out;
    }

    template <class T>
    std::vector<std::shared_ptr<const BlockTable>> tables(const CompareShape &shape,
                                                          const std::vector<T> &threshold,
                                                          unsigned parties,
                                                          Random &random) {
        Bytes secret(bits_bytes(shape.table_bits() * threshold.size()));
        make_comparison_table<T>(shape, threshold, secret);
        const auto shares = split_bits(secret, parties, random);
        std::vector<std::shared_ptr<const BlockTable>> result;
        for (unsigned p = 0; p < parties; ++p) {
            result.push_back(std::make_shared<BlockTable>(shape, threshold.size(), shares[p]));
        }
        return result;
    }

    std::vector<Bytes>
    evals(const CompareShape &shape, std::size_t n, unsigned parties, Random &random) {
        Bytes secret(bits_bytes(shape.eval_bits() * n));
        make_comparison_evaluation(shape, n, secret, [&](auto bytes) {
            random.fill(bytes);
        });
        return split_bits(secret, parties, random);
    }

    class MemoryRound {
        std::vector<Step *> steps_;
        std::vector<RoundPlan> plans_;
        std::vector<Bytes> messages_;
        Bytes opened_;
        std::optional<OpenBuf> buf_;

    public:
        explicit MemoryRound(std::span<Step *const> steps)
            : steps_(steps.begin(), steps.end()), plans_(steps.size()), messages_(steps.size()) {
            for (std::size_t p = 0; p < steps.size(); ++p) {
                steps[p]->bind(plans_[p]);
                plans_[p].seal();
                messages_[p].resize(plans_[p].layout().bytes());
                if (p && plans_[p].layout().fingerprint() != plans_[0].layout().fingerprint()) {
                    throw std::logic_error("test layouts differ");
                }
            }
            opened_.resize(messages_[0].size());
            buf_.emplace(plans_[0].layout(), messages_[0], opened_);
        }

        std::size_t bits() const {
            return plans_[0].layout().bits();
        }

        std::span<const std::byte> opened() const {
            return opened_;
        }

        void run() {
            for (std::size_t p = 0; p < steps_.size(); ++p) {
                OpenWriter w(plans_[p], messages_[p]);
                steps_[p]->contribute(w);
            }
            buf_->initialize();
            for (std::size_t p = 1; p < steps_.size(); ++p) {
                for (std::size_t c = 0; c < plans_[0].layout().chunks().size(); ++c) {
                    const auto &chunk = plans_[0].layout().chunks()[c];
                    buf_->accumulate(c,
                                     std::span<const std::byte>(messages_[p])
                                         .subspan(chunk.offset, chunk.bytes));
                }
            }
            for (std::size_t p = 0; p < steps_.size(); ++p) {
                OpenReader r(plans_[p], opened_);
                steps_[p]->consume(r);
            }
        }
    };

    void check(const std::vector<std::unique_ptr<BitVec>> &out,
               std::span<const unsigned char> expected) {
        for (std::size_t i = 0; i < expected.size(); ++i) {
            unsigned actual = 0;
            for (const auto &p : out) {
                actual ^= p->get(i);
            }
            ASSERT_EQ(actual, expected[i]) << "instance " << i;
        }
    }

    TEST(C1, UniqueProductsAndExactCosts) {
        for (unsigned b = 1; b <= 11; ++b) {
            CompareShape s(b, 1);
            EXPECT_EQ(s.products.size(),
                      (std::size_t{1} << (b - 1)) + (std::size_t{1} << b) - 2 * b - 1);
        }
        CompareShape lt(31, 5);
        EXPECT_EQ(lt.material_bits(), 414);
        EXPECT_EQ(lt.inputs, 13);
        for (auto [k, known, fresh, online, d] :
             std::vector<std::array<unsigned, 5>>{{22, 191, 213, 11, 4},
                                                  {32, 415, 447, 13, 5},
                                                  {49, 895, 944, 15, 6},
                                                  {64, 1919, 1983, 17, 7},
                                                  {72, 3071, 3143, 17, 8}}) {
            EXPECT_EQ(best_compare_d(k - 1), d);
            CompareShape s(k - 1, d);
            EXPECT_EQ(s.material_bits() + 1, known);
            EXPECT_EQ(s.material_bits() + 1 + k, fresh);
            EXPECT_EQ(s.inputs, online);
        }
        for (auto [k, pre, online] :
             std::vector<std::array<unsigned, 3>>{{32, 351, 7}, {64, 2303, 10}}) {
            CompareShape s(k, best_compare_d(k, true), true);
            EXPECT_EQ(s.material_bits(), pre);
            EXPECT_EQ(s.inputs, online);
        }
        for (auto [k, pre, fresh, online] :
             std::vector<std::array<unsigned, 4>>{{32, 1243, 1307, 39}, {64, 7291, 7419, 45}}) {
            CompareShape s(k, best_compare_d(k));
            EXPECT_EQ(3 * s.material_bits() + 1, pre);
            EXPECT_EQ(3 * s.material_bits() + 1 + 2 * k, fresh);
            EXPECT_EQ(3 * s.inputs, online);
        }
    }

    TEST(C1, LTBitsExhaustiveBothDirections) {
        omp_set_num_threads(2);
        Random random;
        for (unsigned ell = 1; ell <= 6; ++ell) {
            for (unsigned d = 1; d <= std::min(ell, 3u); ++d) {
                for (unsigned parties : {2u, 3u}) {
                    const unsigned M = 1u << ell;
                    const auto n = M * M;
                    std::vector<std::uint32_t> c(n), R(n);
                    for (unsigned i = 0; i < n; ++i) {
                        c[i] = i / M;
                        R[i] = i % M;
                    }
                    CompareShape shape(ell, d);
                    const auto tab = tables(shape, c, parties, random);
                    for (bool reverse : {false, true}) {
                        const auto e = evals(shape, n, parties, random);
                        std::vector<std::unique_ptr<BitVec>> out;
                        std::vector<std::unique_ptr<TableCompareStep<std::uint32_t>>> ops;
                        std::vector<Step *> pointers;
                        for (unsigned p = 0; p < parties; ++p) {
                            out.push_back(std::make_unique<BitVec>(n));
                            ops.push_back(std::make_unique<TableCompareStep<std::uint32_t>>(
                                tab[p], e[p], p, R, out[p]->words(), reverse));
                            pointers.push_back(ops.back().get());
                        }
                        MemoryRound round(pointers);
                        round.run();
                        EXPECT_EQ(round.bits(), shape.inputs * n);
                        std::vector<unsigned char> expected(n);
                        for (unsigned i = 0; i < n; ++i) {
                            expected[i] = reverse ? R[i] < c[i] : c[i] < R[i];
                        }
                        check(out, expected);
                    }
                }
            }
        }
    }

    TEST(C1, MSBExhaustiveAllInputsAndMasks) {
        Random random;
        for (unsigned k = 1; k <= 7; ++k) {
            for (unsigned parties : {2u, 3u}) {
                const unsigned M = 1u << k, n = M * M, mask = M - 1;
                std::vector<std::uint32_t> x(n), r(n), hat(n), c(n);
                Bytes sign(bits_bytes(n));
                for (unsigned i = 0; i < n; ++i) {
                    x[i] = i / M;
                    r[i] = i % M;
                    hat[i] = (x[i] + r[i]) & mask;
                    const auto negative = (0 - r[i]) & mask;
                    store_packed(sign, i, 1, negative >> (k - 1));
                    c[i] = k > 1 ? ((M / 2 - 1) - (negative & (M / 2 - 1))) : 0;
                }
                auto sign_shares = split_bits(sign, parties, random);
                std::optional<CompareShape> shape;
                std::vector<std::shared_ptr<const BlockTable>> tab(parties);
                if (k > 1) {
                    shape.emplace(k - 1, std::min(2u, k - 1));
                    tab = tables(*shape, c, parties, random);
                }
                for (bool fresh : {false, true}) {
                    const auto e =
                        shape ? evals(*shape, n, parties, random) : std::vector<Bytes>(parties);
                    auto xs = split_ring<std::uint32_t>(x, k, parties, random),
                         rs = split_ring<std::uint32_t>(r, k, parties, random);
                    std::vector<std::vector<std::uint32_t>> public_hat(
                        parties, fresh ? std::vector<std::uint32_t>(n) : hat);
                    std::vector<std::unique_ptr<BitVec>> out;
                    std::vector<std::unique_ptr<KnownMsbStep<std::uint32_t>>> ops;
                    std::vector<std::unique_ptr<MaskedRingOpen<std::uint32_t>>> openings;
                    std::vector<Step *> pointers, first;
                    for (unsigned p = 0; p < parties; ++p) {
                        out.push_back(std::make_unique<BitVec>(n));
                        ops.push_back(std::make_unique<KnownMsbStep<std::uint32_t>>(
                            k, p, n, public_hat[p], out[p]->words(), tab[p], e[p], sign_shares[p]));
                        pointers.push_back(ops.back().get());
                        if (fresh) {
                            openings.push_back(std::make_unique<MaskedRingOpen<std::uint32_t>>(
                                k, xs[p], rs[p], public_hat[p]));
                            first.push_back(openings.back().get());
                        }
                    }
                    if (fresh) {
                        MemoryRound round(first);
                        round.run();
                        for (const auto &h : public_hat) {
                            EXPECT_EQ(h, hat);
                        }
                    }
                    MemoryRound second(pointers);
                    second.run();
                    std::vector<unsigned char> expected(n);
                    for (unsigned i = 0; i < n; ++i) {
                        expected[i] = x[i] >> (k - 1);
                    }
                    check(out, expected);
                }
            }
        }
    }

    TEST(C1, EQExhaustiveAllErrorsAndMasks) {
        Random random;
        for (unsigned k = 1; k <= 6; ++k) {
            for (unsigned parties : {2u, 3u}) {
                const unsigned M = 1u << k, n = M * M;
                std::vector<std::uint32_t> mask(n), hat(n);
                std::vector<unsigned char> expected(n);
                for (unsigned i = 0; i < n; ++i) {
                    mask[i] = i % M;
                    hat[i] = (i / M + mask[i]) & (M - 1);
                    expected[i] = i / M == 0;
                }
                CompareShape shape(k, std::min(k, 2u), true);
                auto tab = tables(shape, mask, parties, random);
                auto e = evals(shape, n, parties, random);
                std::vector<std::unique_ptr<BitVec>> out;
                std::vector<std::unique_ptr<TableCompareStep<std::uint32_t>>> ops;
                std::vector<Step *> pointers;
                for (unsigned p = 0; p < parties; ++p) {
                    out.push_back(std::make_unique<BitVec>(n));
                    ops.push_back(std::make_unique<TableCompareStep<std::uint32_t>>(
                        tab[p], e[p], p, hat, out[p]->words()));
                    pointers.push_back(ops.back().get());
                }
                MemoryRound round(pointers);
                round.run();
                check(out, expected);
            }
        }
    }

    TEST(C1, LTSExhaustiveAllInputsAndBothMasks) {
        Random random;
        for (unsigned k = 1; k <= 4; ++k) {
            for (unsigned parties : {2u, 3u}) {
                const unsigned M = 1u << k, mask = M - 1, n = M * M * M * M;
                std::vector<std::uint32_t> x(n), y(n), r(n), rp(n), sum(n), dx(n), dy(n);
                Bytes sigma(bits_bytes(n));
                std::vector<unsigned char> expected(n);
                for (unsigned i = 0; i < n; ++i) {
                    unsigned v = i;
                    rp[i] = v % M;
                    v /= M;
                    r[i] = v % M;
                    v /= M;
                    y[i] = v % M;
                    v /= M;
                    x[i] = v;
                    sum[i] = (r[i] + rp[i]) & mask;
                    store_packed(sigma, i, 1, r[i] + rp[i] >= M);
                    dx[i] = (x[i] - rp[i]) & mask;
                    dy[i] = (y[i] + r[i]) & mask;
                    expected[i] = x[i] < y[i];
                }
                CompareShape shape(k, std::min(k, 2u));
                std::array<std::vector<std::shared_ptr<const BlockTable>>, 3> tab = {
                    tables(shape, r, parties, random),
                    tables(shape, rp, parties, random),
                    tables(shape, sum, parties, random)};
                auto sigma_shares = split_bits(sigma, parties, random);
                for (bool fresh : {false, true}) {
                    std::array<std::vector<Bytes>, 3> e = {evals(shape, n, parties, random),
                                                           evals(shape, n, parties, random),
                                                           evals(shape, n, parties, random)};
                    auto xs = split_ring<std::uint32_t>(x, k, parties, random),
                         ys = split_ring<std::uint32_t>(y, k, parties, random),
                         rs = split_ring<std::uint32_t>(r, k, parties, random),
                         rps = split_ring<std::uint32_t>(rp, k, parties, random);
                    std::vector<std::vector<std::uint32_t>> a(parties,
                                                              std::vector<std::uint32_t>(n)),
                        b = a;
                    std::vector<std::unique_ptr<BitVec>> out;
                    std::vector<std::unique_ptr<LtsCompareStep<std::uint32_t>>> ops;
                    std::vector<std::unique_ptr<MaskedRingOpen<std::uint32_t>>> openings;
                    std::vector<Step *> pointers, first;
                    for (unsigned p = 0; p < parties; ++p) {
                        LtsMaterial m;
                        m.width = k;
                        m.count = n;
                        m.party = p;
                        m.stored = !fresh;
                        m.sigma = sigma_shares[p];
                        for (unsigned j = 0; j < 3; ++j) {
                            m.tables[j] = tab[j][p];
                            m.evaluations[j] = e[j][p];
                        }
                        out.push_back(std::make_unique<BitVec>(n));
                        ops.push_back(std::make_unique<LtsCompareStep<std::uint32_t>>(
                            m,
                            fresh ? std::span<const std::uint32_t>(xs[p])
                                  : std::span<const std::uint32_t>(dx),
                            fresh ? std::span<const std::uint32_t>(ys[p])
                                  : std::span<const std::uint32_t>(dy),
                            a[p],
                            b[p],
                            out[p]->words()));
                        pointers.push_back(ops.back().get());
                        if (fresh) {
                            openings.push_back(std::make_unique<MaskedRingOpen<std::uint32_t>>(
                                k, xs[p], rs[p], a[p], ys[p], rps[p], b[p], true));
                            first.push_back(openings.back().get());
                        }
                    }
                    if (fresh) {
                        MemoryRound round(first);
                        round.run();
                    }
                    MemoryRound second(pointers);
                    second.run();
                    check(out, expected);
                }
            }
        }
    }

    TEST(C1, ReusedRhoNegativeControl) {
        Random random;
        constexpr unsigned parties = 2, n = 137;
        CompareShape shape(6, 2);
        std::vector<std::uint32_t> c(n), R(n), Rp(n);
        for (unsigned i = 0; i < n; ++i) {
            c[i] = i % 64;
            R[i] = (i * 3) % 64;
            Rp[i] = (i * 7 + 1) % 64;
        }
        auto tab = tables(shape, c, parties, random);
        auto same = evals(shape, n, parties, random);
        std::vector<std::unique_ptr<BitVec>> out;
        std::vector<std::unique_ptr<TableCompareStep<std::uint32_t>>> a, b;
        std::vector<Step *> pa, pb;
        for (unsigned p = 0; p < parties; ++p) {
            out.push_back(std::make_unique<BitVec>(n));
            a.push_back(std::make_unique<TableCompareStep<std::uint32_t>>(
                tab[p], same[p], p, R, out[p]->words()));
            b.push_back(std::make_unique<TableCompareStep<std::uint32_t>>(
                tab[p], same[p], p, Rp, out[p]->words()));
            pa.push_back(a.back().get());
            pb.push_back(b.back().get());
        }
        MemoryRound first(pa), second(pb);
        first.run();
        second.run();
        unsigned differences = 0;
        for (unsigned i = 0; i < n; ++i) {
            for (unsigned u = 0; u < shape.inputs; ++u) {
                auto bit = [&](unsigned value) {
                    const auto block = u < shape.blocks - 1 ? u : u - (shape.blocks - 1);
                    const auto cb = (c[i] >> ((shape.blocks - 1 - block) * shape.d)) & 3;
                    const auto rb = (value >> ((shape.blocks - 1 - block) * shape.d)) & 3;
                    return u < shape.blocks - 1 ? cb == rb : cb < rb;
                };
                const auto difference = bit(R[i]) ^ bit(Rp[i]);
                EXPECT_EQ(load_packed(first.opened(), u * n + i, 1) ^
                              load_packed(second.opened(), u * n + i, 1),
                          difference);
                differences += difference;
            }
        }
        EXPECT_GT(differences, 0);
    }

    TEST(C1, KnownMsb64NoOnlineAllocations) {
        omp_set_num_threads(2);
        Random random;
        constexpr std::size_t n = 16417;
        constexpr unsigned parties = 3;
        std::vector<std::uint64_t> x(n), r(n), hat(n), c(n);
        random.fill(std::as_writable_bytes(std::span(x)));
        random.fill(std::as_writable_bytes(std::span(r)));
        Bytes sign(bits_bytes(n));
        for (std::size_t i = 0; i < n; ++i) {
            hat[i] = x[i] + r[i];
            const auto neg = 0 - r[i];
            store_packed(sign, i, 1, neg >> 63);
            c[i] = width_mask(63) - (neg & width_mask(63));
        }
        CompareShape shape(63, 7);
        auto tab = tables(shape, c, parties, random);
        auto e = evals(shape, n, parties, random);
        auto signs = split_bits(sign, parties, random);
        std::vector<std::unique_ptr<BitVec>> out;
        std::vector<std::unique_ptr<KnownMsbStep<std::uint64_t>>> ops;
        std::vector<Step *> ptrs;
        for (unsigned p = 0; p < parties; ++p) {
            out.push_back(std::make_unique<BitVec>(n));
            ops.push_back(std::make_unique<KnownMsbStep<std::uint64_t>>(
                64, p, n, hat, out[p]->words(), tab[p], e[p], signs[p]));
            ptrs.push_back(ops.back().get());
        }
        MemoryRound round(ptrs);
        test::allocations = 0;
        test::count_allocations = true;
        round.run();
        test::count_allocations = false;
        EXPECT_EQ(test::allocations.load(), 0);
        std::vector<unsigned char> expected(n);
        ref::msb<std::uint64_t>(64, x, expected);
        check(out, expected);
    }

    TEST(C1, KnownMsb72NoOnlineAllocations) {
        omp_set_num_threads(2);
        Random random;
        constexpr std::size_t n = 16417;
        constexpr unsigned parties = 3;
        std::vector<uint128_t> x(n), r(n), hat(n), c(n);
        random.fill(std::as_writable_bytes(std::span(x)));
        random.fill(std::as_writable_bytes(std::span(r)));
        Bytes sign(bits_bytes(n));
        for (std::size_t i = 0; i < n; ++i) {
            x[i] &= native_mask<uint128_t>(72);
            r[i] &= native_mask<uint128_t>(72);
            hat[i] = (x[i] + r[i]) & native_mask<uint128_t>(72);
            const auto neg = (uint128_t{0} - r[i]) & native_mask<uint128_t>(72);
            store_packed(sign, i, 1, static_cast<std::uint64_t>(neg >> 71));
            c[i] = native_mask<uint128_t>(71) - (neg & native_mask<uint128_t>(71));
        }
        CompareShape shape(71, 8);
        auto tab = tables(shape, c, parties, random);
        auto e = evals(shape, n, parties, random);
        auto signs = split_bits(sign, parties, random);
        std::vector<std::unique_ptr<BitVec>> out;
        std::vector<std::unique_ptr<KnownMsbStep<uint128_t>>> ops;
        std::vector<Step *> ptrs;
        for (unsigned p = 0; p < parties; ++p) {
            out.push_back(std::make_unique<BitVec>(n));
            ops.push_back(std::make_unique<KnownMsbStep<uint128_t>>(
                72, p, n, hat, out[p]->words(), tab[p], e[p], signs[p]));
            ptrs.push_back(ops.back().get());
        }
        MemoryRound round(ptrs);
        test::allocations = 0;
        test::count_allocations = true;
        round.run();
        test::count_allocations = false;
        EXPECT_EQ(test::allocations.load(), 0);
        std::vector<unsigned char> expected(n);
        ref::msb<uint128_t>(72, x, expected);
        check(out, expected);
    }

    TEST(Baseline, EdaExhaustiveAndNoOnlineAllocations) {
        omp_set_num_threads(2);
        Random random;
        for (unsigned k = 1; k <= 8; ++k) {
            for (bool eq : {false, true}) {
                const auto count = std::size_t{1} << k;
                MaterialPlan p;
                p.n = 3;
                p.inputs.push_back({"x", 0, k, count});
                CompareSpec c;
                c.name = "eda";
                c.kind = eq ? CompareKind::EdaEq : CompareKind::EdaMsb;
                c.width = k;
                c.count = count;
                c.inputs = {0};
                p.comparisons.push_back(c);
                char pattern[] = "/tmp/bj-eda-XXXXXX";
                const auto *dir = ::mkdtemp(pattern);
                ASSERT_NE(dir, nullptr);
                const std::filesystem::path path = dir;
                generate_materials(p, MaterialMode::Random, path / "mat");
                std::vector<std::uint32_t> x(count);
                for (unsigned i = 0; i < count; ++i) {
                    x[i] = i;
                }
                const auto xs = split_ring<std::uint32_t>(x, k, 3, random);
                std::vector<std::unique_ptr<MaterialStore>> stores;
                std::vector<std::unique_ptr<EdaBatch<std::uint32_t>>> ops;
                std::vector<std::unique_ptr<BitVec>> out;
                for (unsigned party = 0; party < 3; ++party) {
                    stores.push_back(std::make_unique<MaterialStore>(
                        p, party, path / "mat" / ("party_" + std::to_string(party) + ".bin")));
                    stores.back()->preload();
                    out.push_back(std::make_unique<BitVec>(count));
                    ops.push_back(std::make_unique<EdaBatch<std::uint32_t>>(
                        *stores.back(), p, 0, xs[party], out.back()->words()));
                }
                std::vector<std::unique_ptr<MemoryRound>> rounds;
                for (std::size_t l = 0; l < ops[0]->stage_count(); ++l) {
                    std::array<Step *, 3> steps;
                    for (unsigned party = 0; party < 3; ++party) {
                        steps[party] = &ops[party]->stage(l);
                    }
                    rounds.push_back(std::make_unique<MemoryRound>(steps));
                }
                test::allocations = 0;
                test::count_allocations = true;
                for (auto &round : rounds) {
                    round->run();
                }
                test::count_allocations = false;
                EXPECT_EQ(test::allocations.load(), 0);
                std::vector<unsigned char> expected(count);
                for (unsigned i = 0; i < count; ++i) {
                    expected[i] = eq ? (i == 0) : (i >> (k - 1));
                }
                check(out, expected);
                EXPECT_THROW((EdaBatch<std::uint32_t>(*stores[0], p, 0, xs[0], out[0]->words())),
                             std::logic_error);
                std::error_code error;
                std::filesystem::remove_all(path, error);
            }
        }
    }

    TEST(C1, WidePackingPreservesAdjacentBits) {
        for (unsigned k = 65; k <= 72; ++k) {
            for (unsigned offset = 0; offset < 8; ++offset) {
                std::array<std::byte, 32> data;
                data.fill(std::byte{0xa5});
                const auto before = data;
                const auto x = ((uint128_t{0xde} << 64) | UINT64_C(0xfedcba9876543210)) &
                               native_mask<uint128_t>(k);
                store_wide(data, offset, k, x);
                EXPECT_TRUE(load_wide(data, offset, k) == x);
                for (unsigned bit = 0; bit < 256; ++bit) {
                    if (bit < offset || bit >= offset + k) {
                        EXPECT_EQ(load_packed(data, bit, 1), load_packed(before, bit, 1));
                    }
                }
            }
        }
        static_assert(Ring<72>::msb(uint128_t{1} << 71));
        static_assert(Ring<72>::add(Ring<72>::mask(), 1) == 0);
        static_assert(Ring<72>::reduce<49>(Ring<72>::mask()) == Ring<49>::mask());
    }

    TEST(C1, Fresh72OpeningAcrossChunksNoAllocations) {
        constexpr std::size_t count = 16417;
        Random random;
        std::vector<uint128_t> x(count), r(count), expected(count);
        random.fill(std::as_writable_bytes(std::span(x)));
        random.fill(std::as_writable_bytes(std::span(r)));
        for (std::size_t i = 0; i < count; ++i) {
            x[i] &= Ring<72>::mask();
            r[i] &= Ring<72>::mask();
            expected[i] = Ring<72>::add(x[i], r[i]);
        }
        const auto xs = split_ring<uint128_t>(x, 72, 3, random),
                   rs = split_ring<uint128_t>(r, 72, 3, random);
        std::array<std::vector<uint128_t>, 3> out;
        std::vector<std::unique_ptr<MaskedRingOpen<uint128_t>>> batches;
        std::vector<Step *> steps;
        for (unsigned p = 0; p < 3; ++p) {
            out[p].resize(count);
            batches.push_back(
                std::make_unique<MaskedRingOpen<uint128_t>>(72, xs[p], rs[p], out[p]));
            steps.push_back(batches.back().get());
        }
        MemoryRound round(steps);
        test::allocations = 0;
        test::count_allocations = true;
        round.run();
        test::count_allocations = false;
        EXPECT_EQ(test::allocations.load(), 0);
        for (unsigned p = 0; p < 3; ++p) {
            EXPECT_EQ(out[p], expected);
        }
    }

}
