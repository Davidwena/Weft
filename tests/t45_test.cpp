#include "bj/proto/interval.hpp"
#include "bj/proto/bandpair.hpp"
#include "bj/proto/shuffle.hpp"
#include "bj/ops/compact.hpp"
#include "bj/ref/band.hpp"
#include "allocation.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>
#include <barrier>
#include <exception>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <omp.h>

namespace {
    using namespace bj;

    struct Temp45 {
        std::filesystem::path path;

        Temp45() {
            char name[] = "/tmp/bj-t45-XXXXXX";
            auto p = ::mkdtemp(name);
            if (!p) {
                throw std::runtime_error("mkdtemp");
            }
            path = p;
        }

        ~Temp45() {
            std::error_code e;
            std::filesystem::remove_all(path, e);
        }
    };

    class TestLayer : public Step {
        std::vector<Step *> list_;

    public:
        explicit TestLayer(std::span<Step *const> p) : list_(p.begin(), p.end()) {
        }

        void bind(RoundPlan &p) override {
            for (auto *s : list_) {
                s->bind(p);
            }
        }

        void contribute(OpenWriter &w) override {
            for (auto *s : list_) {
                s->contribute(w);
            }
        }

        void consume(OpenReader &r) override {
            for (auto *s : list_) {
                s->consume(r);
            }
        }
    };

    class Memory45 {
        std::vector<Step *> steps_;
        std::vector<RoundPlan> plans_;
        std::vector<std::vector<std::byte>> messages_;
        std::vector<std::byte> opened_;
        std::optional<OpenBuf> buf_;

    public:
        explicit Memory45(std::span<Step *const> steps)
            : steps_(steps.begin(), steps.end()), plans_(steps.size()), messages_(steps.size()) {
            for (unsigned p = 0; p < steps.size(); ++p) {
                steps[p]->bind(plans_[p]);
                plans_[p].seal();
                messages_[p].resize(plans_[p].layout().bytes());
                if (p && plans_[p].layout().fingerprint() != plans_[0].layout().fingerprint()) {
                    throw std::logic_error("different layouts");
                }
            }
            opened_.resize(messages_[0].size());
            buf_.emplace(plans_[0].layout(), messages_[0], opened_);
        }

        void run() {
            for (unsigned p = 0; p < steps_.size(); ++p) {
                OpenWriter w(plans_[p], messages_[p]);
                steps_[p]->contribute(w);
            }
            buf_->initialize();
            for (unsigned p = 1; p < steps_.size(); ++p) {
                for (std::size_t j = 0; j < plans_[0].layout().chunks().size(); ++j) {
                    const auto &c = plans_[0].layout().chunks()[j];
                    buf_->accumulate(
                        j, std::span<const std::byte>(messages_[p]).subspan(c.offset, c.bytes));
                }
            }
            for (unsigned p = 0; p < steps_.size(); ++p) {
                OpenReader r(plans_[p], opened_);
                steps_[p]->consume(r);
            }
        }
    };

    std::vector<std::unique_ptr<MaterialStore>> stores_for(const MaterialPlan &p,
                                                           const std::filesystem::path &d) {
        std::vector<std::unique_ptr<MaterialStore>> stores;
        for (unsigned party = 0; party < p.n; ++party) {
            stores.push_back(std::make_unique<MaterialStore>(
                p, party, d / ("party_" + std::to_string(party) + ".bin")));
            stores.back()->preload();
        }
        return stores;
    }

    void fill_masked(MaskedColumn &col,
                     std::size_t input,
                     unsigned party,
                     const MaterialPlan &p,
                     const std::vector<std::unique_ptr<MaterialStore>> &stores,
                     std::span<const uint128_t> values) {
        auto owner = p.inputs[input].owner;
        const auto full = stores[owner]->packed(stores[owner]->layout().owner_masks[input]),
                   share = stores[party]->packed(stores[party]->layout().input_shares[input]);
        auto fill = [&]<class T>() {
            unpack_native<T>(col.width, share, col.mask<T>());
            const auto mask = native_mask<T>(col.width);
            for (std::size_t i = 0; i < values.size(); ++i) {
                col.delta<T>()[i] =
                    static_cast<T>(values[i] + load_wide(full, i * col.width, col.width)) & mask;
            }
        };
        if (col.width <= 32) {
            fill.template operator()<std::uint32_t>();
        } else if (col.width <= 64) {
            fill.template operator()<std::uint64_t>();
        } else {
            fill.template operator()<uint128_t>();
        }
    }

    TEST(C2, IntervalExhaustiveValidSignedRangeAndSharedOffsets) {
        for (unsigned k : {6u, 7u}) {
            Temp45 temp;
            MaterialPlan p;
            p.n = 3;
            const auto N = std::size_t{1} << k;
            p.inputs.push_back({"masked", 0, k, N});
            CompareSpec spec;
            spec.name = "interval";
            spec.kind = CompareKind::Interval;
            spec.width = k;
            spec.count = N;
            spec.inputs = {0};
            spec.evaluations = 4;
            p.comparisons.push_back(spec);
            generate_materials(p, MaterialMode::Random, temp.path / "m");
            auto stores = stores_for(p, temp.path / "m");
            const auto full = stores[0]->packed(stores[0]->layout().owner_masks[0]);
            std::vector<std::uint32_t> hat(N);
            for (std::size_t i = 0; i < N; ++i) {
                hat[i] = (i + load_packed(full, i * k, k)) & width_mask(k);
            }
            const std::array<BandWindow, 3> windows{{{0, 0}, {0, 2}, {1, 3}}};
            std::vector<std::vector<BitVec>> output;
            std::vector<std::unique_ptr<Interval<std::uint32_t>>> steps;
            std::vector<Step *> pointers;
            for (unsigned party = 0; party < 3; ++party) {
                output.emplace_back();
                for (unsigned w = 0; w < 3; ++w) {
                    output.back().emplace_back(N);
                }
                std::array<std::span<std::uint64_t>, 3> outs;
                for (unsigned w = 0; w < 3; ++w) {
                    outs[w] = output.back()[w].words();
                }
                steps.push_back(std::make_unique<Interval<std::uint32_t>>(
                    *stores[party], p, 0, hat, windows, outs));
                pointers.push_back(steps.back().get());
            }
            Memory45 round(pointers);
            test::allocations = 0;
            test::count_allocations = true;
            round.run();
            test::count_allocations = false;
            EXPECT_EQ(test::allocations.load(), 0);
            for (unsigned w = 0; w < 3; ++w) {
                for (std::size_t i = 0; i < N; ++i) {
                    const auto g =
                        i < N / 2 ? static_cast<std::int64_t>(i)
                                  : static_cast<std::int64_t>(i) - static_cast<std::int64_t>(N);
                    const auto low = g - static_cast<std::int64_t>(windows[w].lower),
                               high = g - static_cast<std::int64_t>(windows[w].upper) - 1;
                    if (low < -static_cast<std::int64_t>(N / 2) ||
                        high < -static_cast<std::int64_t>(N / 2)) {
                        continue;
                    }
                    unsigned bit = 0;
                    for (unsigned party = 0; party < 3; ++party) {
                        bit ^= output[party][w].get(i);
                    }
                    EXPECT_EQ(bit,
                              g >= static_cast<std::int64_t>(windows[w].lower) &&
                                  g <= static_cast<std::int64_t>(windows[w].upper));
                }
            }
        }
    }

    TEST(C2, AmountAllPairsAndFiveVariants) {
        omp_set_num_threads(2);
        for (const auto &variant : {"C1", "C1+zero", "+interval", "+amount", "C1+C2"}) {
            Temp45 temp;
            InvestigationConfig c;
            c.mode = "band";
            c.n = 3;
            c.rows = {0, 0, 0};
            c.kin = c.kout = 24;
            c.bt = 3;
            c.ba = 5;
            c.bq = 3;
            c.kt = 6;
            c.ka = 10;
            c.p = 1;
            c.q = 4;
            c.windows = {{0, 2}};
            c.variant = variant;
            auto p = compile_investigation(c);
            generate_materials(p, MaterialMode::Random, temp.path / "m");
            auto stores = stores_for(p, temp.path / "m");
            const std::array<unsigned, 3> widths{64, 96, 0};
            std::vector<std::unique_ptr<MaskedTable>> in, out;
            std::vector<std::unique_ptr<BandPair>> batches;
            std::vector<uint128_t> t(24, 3), a(24);
            for (unsigned i = 0; i < 24; ++i) {
                a[i] = i;
            }
            for (unsigned party = 0; party < 3; ++party) {
                in.push_back(std::make_unique<MaskedTable>(24, widths));
                out.push_back(std::make_unique<MaskedTable>(24, widths));
                fill_masked(in.back()->columns[0], named_mask(p, "tin"), party, p, stores, t);
                fill_masked(out.back()->columns[0], named_mask(p, "tout"), party, p, stores, t);
                fill_masked(in.back()->columns[1], named_mask(p, "ain"), party, p, stores, a);
                fill_masked(out.back()->columns[1], named_mask(p, "aout"), party, p, stores, a);
                batches.push_back(make_band_pair(c, *stores[party], p, *in.back(), *out.back()));
                batches.back()->prepare();
            }
            auto run = [&](unsigned stage) {
                std::vector<std::unique_ptr<TestLayer>> layers;
                std::vector<Step *> ptrs;
                for (auto &batch : batches) {
                    auto list = stage == 0   ? batch->opening_steps()
                                : stage == 1 ? batch->comparison_steps()
                                             : batch->product_steps();
                    layers.push_back(std::make_unique<TestLayer>(list));
                    ptrs.push_back(layers.back().get());
                }
                Memory45 round(ptrs);
                test::allocations = 0;
                test::count_allocations = true;
                round.run();
                test::count_allocations = false;
                EXPECT_EQ(test::allocations.load(), 0);
            };
            if (!batches[0]->opening_steps().empty()) {
                run(0);
            }
            run(1);
            for (auto &batch : batches) {
                batch->combine();
            }
            run(2);
            unsigned matches = 0;
            for (unsigned i = 0; i < 24; ++i) {
                for (unsigned j = 0; j < 24; ++j) {
                    std::uint64_t result = 0;
                    for (auto &batch : batches) {
                        result += batch->shares(0)[i * 24 + j];
                    }
                    const bool expected = 4 * j <= 5 * i && 3 * i <= 4 * j;
                    EXPECT_EQ(result, expected) << variant << " amounts " << i << "," << j;
                    matches += expected;
                }
            }
            EXPECT_GT(matches, 0);
            const auto roundtrip = MaterialPlan::load(temp.path / "m/plan.json");
            EXPECT_EQ(roundtrip.fingerprint(), p.fingerprint());
        }
    }

    std::vector<Host> local_hosts(unsigned n) {
        std::vector<Host> result;
        std::vector<int> sockets;
        for (unsigned i = 0; i < n; ++i) {
            int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                throw std::runtime_error("socket");
            }
            sockets.push_back(fd);
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr))) {
                throw std::runtime_error("bind");
            }
            socklen_t size = sizeof(addr);
            ::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &size);
            result.push_back({"127.0.0.1", ntohs(addr.sin_port)});
        }
        for (int fd : sockets) {
            ::close(fd);
        }
        return result;
    }

    TEST(Shuffle, RealChainMixedWidthsAndInvalidMaskGuard) {
        for (auto mode : {MaterialMode::Random, MaterialMode::Prg}) {
            Temp45 temp;
            MaterialPlan p;
            p.n = 3;
            constexpr std::size_t N = 73;
            const std::array<unsigned, 4> widths{32, 64, 72, 0};
            for (unsigned c = 0; c < 4; ++c) {
                p.inputs.push_back({"x" + std::to_string(c), c % 3, widths[c], N});
            }
            std::vector<std::size_t> refs;
            for (unsigned c = 0; c < 4; ++c) {
                p.masks.push_back({"out" + std::to_string(c), widths[c], N, true, true, {}});
                refs.push_back(4 + c);
            }
            p.shuffles.push_back({"s", N, {0, 1, 2, 3}, refs});
            generate_materials(p, mode, temp.path / "m");
            auto stores = stores_for(p, temp.path / "m");
            std::vector<std::unique_ptr<MaskedTable>> tables;
            std::vector<std::unique_ptr<Shuffle>> shuffles;
            for (unsigned party = 0; party < 3; ++party) {
                tables.push_back(std::make_unique<MaskedTable>(N, widths));
                for (unsigned col = 0; col < 3; ++col) {
                    std::vector<uint128_t> value(N);
                    for (unsigned i = 0; i < N; ++i) {
                        value[i] = col == 0 ? i : col == 1 ? 3 * i : (uint128_t{1} << 68) + 5 * i;
                    }
                    fill_masked(tables.back()->columns[col], col, party, p, stores, value);
                }
                const auto full = stores[0]->packed(stores[0]->layout().owner_masks[3]),
                           share = stores[party]->packed(stores[party]->layout().input_shares[3]);
                tables.back()->columns[3].mask_bits.unpack(share);
                for (unsigned i = 0; i < N; ++i) {
                    tables.back()->columns[3].bits.set(i, (i & 1) ^ load_packed(full, i, 1));
                }
                shuffles.push_back(std::make_unique<Shuffle>(*tables.back(), *stores[party], p, 0));
            }
            MaskedColumn reused(32, N);
            EXPECT_THROW(load_mask_share(reused, *stores[0], p, 4), std::logic_error);
            Config config;
            config.hosts = local_hosts(3);
            config.threads = 2;
            config.timeout_ms = 5000;
            std::array<std::exception_ptr, 3> errors{};
            std::vector<std::thread> threads;
            for (unsigned party = 0; party < 3; ++party) {
                threads.emplace_back([&, party] {
                    try {
                        Net net(party, config);
                        tables[party]->mark_public_reordered();
                        bool rejected = false;
                        try {
                            shuffles[party]->run(net);
                        } catch (const std::logic_error &) {
                            rejected = true;
                        }
                        if (!rejected) {
                            throw std::runtime_error("invalid-mask guard failed");
                        }
                        tables[party]->dealer_knows_positions = true;
                        for (auto &col : tables[party]->columns) {
                            col.dealer_knows_mask = true;
                        }
                        shuffles[party]->run(net);
                        rejected = false;
                        try {
                            shuffles[party]->run(net);
                        } catch (const std::logic_error &) {
                            rejected = true;
                        }
                        if (!rejected) {
                            throw std::runtime_error("shuffle one-time guard failed");
                        }
                    } catch (...) {
                        errors[party] = std::current_exception();
                    }
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
            for (auto error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
            std::vector<unsigned char> seen(N);
            for (unsigned row = 0; row < N; ++row) {
                std::uint32_t m32 = 0;
                std::uint64_t m64 = 0;
                uint128_t m72 = 0;
                bool bit = false;
                for (unsigned party = 0; party < 3; ++party) {
                    m32 += tables[party]->columns[0].m32.data()[row];
                    m64 += tables[party]->columns[1].m64.data()[row];
                    m72 += tables[party]->columns[2].m128.data()[row];
                    bit ^= tables[party]->columns[3].mask_bits.get(row);
                }
                const auto id = tables[0]->columns[0].d32.data()[row] - m32;
                ASSERT_LT(id, N);
                ASSERT_EQ(seen[id]++, 0);
                EXPECT_EQ(tables[0]->columns[1].d64.data()[row] - m64, 3 * id);
                EXPECT_TRUE(((tables[0]->columns[2].d128.data()[row] - m72) & Ring<72>::mask()) ==
                            (uint128_t{1} << 68) + 5 * id);
                EXPECT_EQ(tables[0]->columns[3].bits.get(row) ^ bit, id & 1);
            }
        }
    }

    TEST(Compact, RetainsExactPublicPrefixRowsAndValidityBits) {
        for (unsigned scenario = 0; scenario < 3; ++scenario) {
            Temp45 temp;
            InvestigationConfig c;
            c.n = 3;
            c.initiator = 2;
            c.rows = {4, 4, 4};
            c.kin = c.kout = 3;
            c.mode = "compact";
            auto p = compile_investigation(c);
            generate_materials(
                p, scenario == 1 ? MaterialMode::Prg : MaterialMode::Random, temp.path / "m");
            auto stores = stores_for(p, temp.path / "m");
            std::vector<ref::PlainRow> plain;
            std::vector<std::unique_ptr<InputManager>> inputs;
            std::vector<std::unique_ptr<Compact>> compact;
            for (unsigned bank = 0; bank < 3; ++bank) {
                nlohmann::json j = {{"inputs", nlohmann::json::object()}};
                const auto suffix = "_" + std::to_string(bank);
                for (auto name : {"sid", "dir", "t", "a"}) {
                    j["inputs"][std::string(name) + suffix] = nlohmann::json::array();
                }
                for (unsigned i = 0; i < 4; ++i) {
                    const unsigned row = bank * 4 + i;
                    bool selected = scenario == 2 || (scenario == 1 && row < 6);
                    ref::PlainRow r;
                    r.subject = selected ? 42 : 1000;
                    r.time = 100 + row;
                    r.amount = 100 + row;
                    r.outgoing = row % 2;
                    plain.push_back(r);
                    j["inputs"]["sid" + suffix].push_back(r.subject);
                    j["inputs"]["dir" + suffix].push_back(r.outgoing);
                    j["inputs"]["t" + suffix].push_back(r.time);
                    j["inputs"]["a" + suffix].push_back(static_cast<std::uint64_t>(r.amount));
                }
                if (bank == 2) {
                    j["inputs"]["subject"] = {42};
                }
                const auto file = temp.path / ("input" + std::to_string(bank) + ".json");
                {
                    std::ofstream f(file);
                    f << j;
                }
                inputs.push_back(std::make_unique<InputManager>(*stores[bank], p, file));
                compact.push_back(std::make_unique<Compact>(c, *stores[bank], p, *inputs.back()));
                stores[bank]->claim_execution();
                stores[bank]->begin_online();
            }
            Config config;
            config.hosts = local_hosts(3);
            config.threads = 2;
            config.timeout_ms = 5000;
            std::vector<std::thread> threads;
            std::array<std::exception_ptr, 3> errors{};
            for (unsigned bank = 0; bank < 3; ++bank) {
                threads.emplace_back([&, bank] {
                    try {
                        Net net(bank, config);
                        inputs[bank]->run(net);
                        compact[bank]->run(net);
                    } catch (...) {
                        errors[bank] = std::current_exception();
                    }
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
            for (auto e : errors) {
                if (e) {
                    std::rethrow_exception(e);
                }
            }
            const auto expected = ref::investigation(plain, 42, c);
            for (unsigned direction = 0; direction < 2; ++direction) {
                std::vector<std::pair<std::uint64_t, std::uint64_t>> actual, want;
                const auto &rows = direction ? expected.outbound : expected.inbound;
                for (auto r : rows) {
                    want.emplace_back(r.time, static_cast<std::uint64_t>(r.amount));
                }
                for (unsigned slot = 0; slot < 3; ++slot) {
                    std::uint64_t tm = 0;
                    uint128_t am = 0;
                    bool bm = false;
                    for (auto &bank : compact) {
                        const auto &table = direction ? bank->outbound() : bank->inbound();
                        tm += table.columns[0].m64.data()[slot];
                        am += table.columns[1].m128.data()[slot];
                        bm ^= table.columns[2].mask_bits.get(slot);
                    }
                    const auto &table = direction ? compact[0]->outbound() : compact[0]->inbound();
                    const auto t = table.columns[0].d64.data()[slot] - tm;
                    const auto a = (table.columns[1].d128.data()[slot] - am) & Ring<72>::mask();
                    const bool valid = table.columns[2].bits.get(slot) ^ bm;
                    if (valid) {
                        actual.emplace_back(t, static_cast<std::uint64_t>(a));
                    } else {
                        EXPECT_EQ(t,
                                  direction ? 0 - (std::uint64_t{1} << c.bt)
                                            : (std::uint64_t{1} << c.bt));
                        EXPECT_TRUE(a == 0);
                    }
                }
                std::sort(actual.begin(), actual.end());
                std::sort(want.begin(), want.end());
                EXPECT_EQ(actual, want);
                bool omega = false;
                for (auto &bank : compact) {
                    omega ^= (direction ? bank->omega_out() : bank->omega_in())[0] & 1;
                }
                EXPECT_EQ(omega, scenario == 2);
            }
            for (auto &bank : compact) {
                EXPECT_EQ(bank->stats().logical_rounds, 14);
            }
        }
    }

    TEST(Shuffle, HonestPermutationUniformityAndNegativeControl) {
        constexpr unsigned count = 7, trials = 6000;
        std::array<unsigned, count> frequency{};
        Prg prg{Prg::Key{}};
        std::array<std::byte, 16> seed{};
        std::array<std::uint32_t, count> permutation{};
        for (unsigned trial = 0; trial < trials; ++trial) {
            prg.fill(2026, trial * 16, seed);
            permutation_from_seed(seed, permutation);
            for (unsigned i = 0; i < count; ++i) {
                if ((permutation[count - 1 - i] + 1) % count == 0) {
                    ++frequency[i];
                }
            }
        }
        double chi = 0;
        const double expected = double(trials) / count;
        for (auto f : frequency) {
            chi += (f - expected) * (f - expected) / expected;
        }
        RecordProperty("chi_square_df6", chi);
        EXPECT_LT(chi, 16.812);
        const double broken =
            (trials - expected) * (trials - expected) / expected + (count - 1) * expected;
        EXPECT_GT(broken, 1000);
    }

    TEST(Plan45, RejectInvalidPublicRanges) {
        InvestigationConfig c;
        c.rows = {7, 7, 7};
        c.kin = c.kout = 4;
        EXPECT_NO_THROW(c.validate());
        auto wrong = c;
        wrong.windows = {{9, 8}};
        EXPECT_THROW(wrong.validate(), std::invalid_argument);
        wrong = c;
        wrong.p = wrong.q;
        EXPECT_THROW(wrong.validate(), std::invalid_argument);
        wrong = c;
        wrong.kt = 26;
        EXPECT_THROW(wrong.validate(), std::invalid_argument);
        wrong = c;
        wrong.kp = 3;
        EXPECT_THROW(wrong.validate(), std::invalid_argument);
        wrong = c;
        wrong.ka = 47;
        EXPECT_THROW(wrong.validate(), std::invalid_argument);
        wrong = c;
        wrong.kin = wrong.kout = 65536;
        EXPECT_THROW(wrong.validate(), std::invalid_argument);
    }
}
