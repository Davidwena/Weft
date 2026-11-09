#include "allocation.hpp"
#include "bj/ops/compact.hpp"
#include "bj/ops/patterns.hpp"
#include "bj/proto/and2a.hpp"
#include "bj/proto/bitmul.hpp"
#include "bj/proto/eq.hpp"
#include "bj/proto/multiply.hpp"
#include "bj/ref/band.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <random>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
    using namespace bj;

    struct Temp {
        std::filesystem::path path;

        Temp() {
            char name[] = "/tmp/bj-v8-XXXXXX";
            const auto p = ::mkdtemp(name);
            if (!p) {
                throw std::runtime_error("mkdtemp");
            }
            path = p;
        }

        ~Temp() {
            std::error_code e;
            std::filesystem::remove_all(path, e);
        }
    };

    class MemoryRound {
        std::vector<std::vector<Step *>> steps_;
        std::vector<RoundPlan> plans_;
        std::vector<std::vector<std::byte>> messages_;
        std::vector<std::byte> opened_;
        std::optional<OpenBuf> buf_;

    public:
        MemoryRound(std::vector<std::vector<Step *>> steps)
            : steps_(std::move(steps)), plans_(steps_.size()), messages_(steps_.size()) {
            for (unsigned p = 0; p < steps_.size(); ++p) {
                for (auto *step : steps_[p]) {
                    step->bind(plans_[p]);
                }
                plans_[p].seal();
                messages_[p].resize(plans_[p].layout().bytes());
                if (p && plans_[p].layout().fingerprint() != plans_[0].layout().fingerprint()) {
                    throw std::logic_error("batch schemas differ");
                }
            }
            opened_.resize(messages_[0].size());
            buf_.emplace(plans_[0].layout(), messages_[0], opened_);
        }

        void run() {
            for (unsigned p = 0; p < steps_.size(); ++p) {
                OpenWriter w(plans_[p], messages_[p]);
                for (auto *step : steps_[p]) {
                    step->contribute(w);
                }
            }
            buf_->initialize();
            for (unsigned p = 1; p < steps_.size(); ++p) {
                for (std::size_t ch = 0; ch < plans_[0].layout().chunks().size(); ++ch) {
                    const auto &c = plans_[0].layout().chunks()[ch];
                    buf_->accumulate(
                        ch, std::span<const std::byte>(messages_[p]).subspan(c.offset, c.bytes));
                }
            }
            for (unsigned p = 0; p < steps_.size(); ++p) {
                OpenReader r(plans_[p], opened_);
                for (auto *step : steps_[p]) {
                    step->consume(r);
                }
            }
        }
    };

    std::vector<std::unique_ptr<MaterialStore>> stores(const MaterialPlan &p,
                                                       const std::filesystem::path &dir) {
        std::vector<std::unique_ptr<MaterialStore>> result;
        for (unsigned i = 0; i < p.n; ++i) {
            result.push_back(std::make_unique<MaterialStore>(
                p, i, dir / ("party_" + std::to_string(i) + ".bin")));
            result.back()->preload();
        }
        return result;
    }

    void masked(MaskedColumn &c,
                std::size_t input,
                unsigned party,
                const MaterialPlan &p,
                const std::vector<std::unique_ptr<MaterialStore>> &store,
                std::span<const uint128_t> values) {
        const auto &spec = p.inputs[input];
        const auto full = store[spec.owner]->packed(store[spec.owner]->layout().owner_masks[input]);
        auto fill = [&]<class T>() {
            unpack_native<T>(spec.width,
                             store[party]->packed(store[party]->layout().input_shares[input]),
                             c.mask<T>());
            for (std::size_t i = 0; i < c.count; ++i) {
                c.delta<T>()[i] =
                    static_cast<T>(values[i] + load_wide(full, i * spec.width, spec.width)) &
                    native_mask<T>(spec.width);
            }
        };
        if (spec.width <= 64) {
            fill.template operator()<std::uint64_t>();
        } else {
            fill.template operator()<uint128_t>();
        }
    }

    TEST(V8, BitMulSmallExhaustionWideAndTripleEquivalence) {
        constexpr std::size_t N = 32;
        for (auto mode : {MaterialMode::Random, MaterialMode::Prg}) {
            Temp tmp;
            MaterialPlan p;
            p.n = 3;
            p.context = "V8 bitmul unit";
            p.inputs = {{"x", 0, 96, N}};
            p.bitmuls = {{"b32", 32, N, 0}, {"b64", 64, N, 0}, {"b96", 96, N, 0}};
            p.gates = {{"b2a", 1, 64, N, {}}};
            p.multiplications = {{"product", 64, N}};
            auto report = generate_materials(p, mode, tmp.path / "m");
            auto ss = stores(p, tmp.path / "m");
            std::vector<uint128_t> value(N);
            for (std::size_t i = 0; i < N; ++i) {
                value[i] = i / 2;
            }
            std::array<std::unique_ptr<MaskedColumn>, 3> col;
            std::array<std::unique_ptr<BitVec>, 3> bit;
            std::array<std::vector<std::uint32_t>, 3> o32;
            std::array<std::vector<std::uint64_t>, 3> o64, arithmetic, xshare, product;
            std::array<std::vector<uint128_t>, 3> o96;
            std::array<std::unique_ptr<BitMul<std::uint32_t>>, 3> m32;
            std::array<std::unique_ptr<BitMul<std::uint64_t>>, 3> m64;
            std::array<std::unique_ptr<BitMul<uint128_t>>, 3> m96;
            std::array<std::unique_ptr<FusedAnd2A<1, std::uint64_t>>, 3> b2a;
            std::array<std::unique_ptr<BeaverProduct<std::uint64_t>>, 3> triple;
            std::vector<std::vector<Step *>> first(3), second(3);
            for (unsigned party = 0; party < 3; ++party) {
                col[party] = std::make_unique<MaskedColumn>(96, N);
                masked(*col[party], 0, party, p, ss, value);
                bit[party] = std::make_unique<BitVec>(N);
                for (std::size_t i = 0; i < N; ++i) {
                    bit[party]->set(i,
                                    party == 0   ? ((i * 13 + 1) % 3 == 0)
                                    : party == 1 ? ((i * 7) % 5 == 0)
                                                 : bool(i % 2) ^ bool((i * 13 + 1) % 3 == 0) ^
                                                       bool((i * 7) % 5 == 0));
                }
                o32[party].resize(N);
                o64[party].resize(N);
                o96[party].resize(N);
                arithmetic[party].resize(N);
                xshare[party].resize(N);
                product[party].resize(N);
                for (std::size_t i = 0; i < N; ++i) {
                    xshare[party][i] = static_cast<std::uint64_t>(
                        (party == 0 ? col[party]->d128.data()[i] : 0) - col[party]->m128.data()[i]);
                }
                m32[party] = std::make_unique<BitMul<std::uint32_t>>(
                    *ss[party], p, 0, bit[party]->words(), *col[party], o32[party]);
                m64[party] = std::make_unique<BitMul<std::uint64_t>>(
                    *ss[party], p, 1, bit[party]->words(), *col[party], o64[party]);
                m96[party] = std::make_unique<BitMul<uint128_t>>(
                    *ss[party], p, 2, bit[party]->words(), *col[party], o96[party]);
                b2a[party] = std::make_unique<FusedAnd2A<1, std::uint64_t>>(
                    *ss[party],
                    p,
                    0,
                    std::array<std::span<const std::uint64_t>, 1>{bit[party]->words()},
                    arithmetic[party]);
                triple[party] = std::make_unique<BeaverProduct<std::uint64_t>>(
                    *ss[party], p, 0, arithmetic[party], xshare[party], product[party]);
                first[party] = {
                    m32[party].get(), m64[party].get(), m96[party].get(), b2a[party].get()};
                second[party] = {triple[party].get()};
                ss[party]->claim_execution();
            }
            MemoryRound a(first), b(second);
            for (auto &store : ss) {
                store->begin_online();
            }
            test::allocations = 0;
            test::count_allocations = true;
            a.run();
            b.run();
            test::count_allocations = false;
            EXPECT_EQ(test::allocations.load(), 0u);
            for (std::size_t i = 0; i < N; ++i) {
                std::uint32_t a32 = 0;
                std::uint64_t a64 = 0, bytriple = 0;
                uint128_t a96 = 0;
                for (unsigned party = 0; party < 3; ++party) {
                    a32 += o32[party][i];
                    a64 += o64[party][i];
                    a96 += o96[party][i];
                    bytriple += product[party][i];
                }
                const auto expected = static_cast<std::uint64_t>((i % 2) * (i / 2));
                EXPECT_EQ(a32, expected);
                EXPECT_EQ(a64, expected);
                EXPECT_EQ(bytriple, expected);
                EXPECT_TRUE((a96 & Ring<96>::mask()) == expected);
            }
            EXPECT_THROW(ss[0]->claim_bitmul(0), std::logic_error);
            EXPECT_THROW(a.run(), std::logic_error);
            const auto &ids = ss[0]->layout().bitmuls;
            for (unsigned j = 0; j < 3; ++j) {
                std::uint64_t bits = 0;
                for (auto id : ids[j]) {
                    bits += ss[0]->layout().segments[id].bits;
                }
                EXPECT_EQ(bits, N * (1 + 2 * p.bitmuls[j].width));
            }
        }
    }

    TEST(V8, FreshMsb96AndFreshEq64RandomBoundaries) {
        constexpr std::size_t N = 73;
        Temp tmp;
        MaterialPlan p;
        p.n = 3;
        p.inputs = {{"x", 0, 96, N}, {"y", 1, 64, N}};
        CompareSpec a;
        a.name = "msb";
        a.kind = CompareKind::MsbFresh;
        a.width = 96;
        a.count = N;
        a.inputs = {0};
        CompareSpec b;
        b.name = "eq";
        b.kind = CompareKind::EqFresh;
        b.width = 64;
        b.count = N;
        b.inputs = {1};
        p.comparisons = {a, b};
        const auto rep = generate_materials(p, MaterialMode::Prg, tmp.path / "m");
        for (auto bits : rep.gate_bits) {
            EXPECT_EQ(bits, N * (8799 + 2367));
        }
        EXPECT_EQ(a.block_width(), 9u);
        auto ss = stores(p, tmp.path / "m");
        std::vector<uint128_t> xv(N), yv(N);
        std::mt19937_64 rng(73911);
        for (std::size_t i = 0; i < N; ++i) {
            xv[i] = ((uint128_t(rng()) << 64) | rng()) & Ring<96>::mask();
            yv[i] = i % 3 ? rng() : 0;
        }
        xv[0] = 0;
        xv[1] = 1;
        xv[2] = Ring<96>::mask();
        xv[3] = uint128_t{1} << 95;
        xv[4] = (uint128_t{1} << 95) - 1;
        std::array<std::unique_ptr<MaskedColumn>, 3> x, y;
        std::array<std::vector<uint128_t>, 3> xs;
        std::array<std::vector<std::uint64_t>, 3> ys;
        std::array<std::unique_ptr<BitVec>, 3> om, oe;
        std::array<std::unique_ptr<MsbFresh<uint128_t>>, 3> msb;
        std::array<std::unique_ptr<EqFresh<std::uint64_t>>, 3> eq;
        std::vector<std::vector<Step *>> first(3), second(3);
        for (unsigned party = 0; party < 3; ++party) {
            x[party] = std::make_unique<MaskedColumn>(96, N);
            y[party] = std::make_unique<MaskedColumn>(64, N);
            masked(*x[party], 0, party, p, ss, xv);
            masked(*y[party], 1, party, p, ss, yv);
            xs[party].resize(N);
            ys[party].resize(N);
            for (std::size_t i = 0; i < N; ++i) {
                xs[party][i] =
                    ((party == 0 ? x[party]->d128.data()[i] : 0) - x[party]->m128.data()[i]) &
                    Ring<96>::mask();
                ys[party][i] = (party == 0 ? y[party]->d64.data()[i] : 0) - y[party]->m64.data()[i];
            }
            om[party] = std::make_unique<BitVec>(N);
            oe[party] = std::make_unique<BitVec>(N);
            msb[party] = std::make_unique<MsbFresh<uint128_t>>(
                *ss[party], p, 0, xs[party], om[party]->words());
            eq[party] = std::make_unique<EqFresh<std::uint64_t>>(
                *ss[party], p, 1, ys[party], oe[party]->words());
            first[party] = {&msb[party]->opening(), &eq[party]->opening()};
            second[party] = {&msb[party]->comparison(), &eq[party]->comparison()};
            ss[party]->claim_execution();
        }
        MemoryRound opening(first), comparison(second);
        for (auto &store : ss) {
            store->begin_online();
        }
        test::allocations = 0;
        test::count_allocations = true;
        opening.run();
        comparison.run();
        test::count_allocations = false;
        EXPECT_EQ(test::allocations.load(), 0u);
        for (std::size_t i = 0; i < N; ++i) {
            bool sign = false, equal = false;
            for (unsigned party = 0; party < 3; ++party) {
                sign ^= om[party]->get(i);
                equal ^= oe[party]->get(i);
            }
            EXPECT_EQ(sign, bool(xv[i] >> 95));
            EXPECT_EQ(equal, yv[i] == 0);
        }
        EXPECT_EQ(MaterialPlan::load(tmp.path / "m/plan.json").fingerprint(), p.fingerprint());
    }

    TEST(V8, NewRangeChecksAndCrossGroupCancellation) {
        InvestigationConfig c;
        c.mode = "band";
        c.rows = {0, 0, 0};
        c.kin = c.kout = 4;
        c.pattern = "P4";
        c.algorithm = "dc";
        EXPECT_NO_THROW(c.validate());
        c.windows = {{0, std::uint64_t{1} << c.bt}};
        EXPECT_THROW(c.validate(), std::invalid_argument);
        c.windows = {{0, 1800}};
        c.kin = c.kout = 1 << 20;
        EXPECT_THROW(c.validate(), std::invalid_argument);
        c.pattern = "P2";
        c.kin = c.kout = 4;
        c.ba = 62;
        EXPECT_THROW(c.validate(), std::invalid_argument);
        c.pattern = "P3";
        c.ba = 40;
        c.amount_lower = 100;
        c.amount_upper = 100;
        EXPECT_THROW(c.validate(), std::invalid_argument);

        const std::int64_t theta = std::int64_t{1} << 25;
        EXPECT_TRUE(ref::band(theta - 1, 100, theta, 100, {0, 1}, 5, 100));
        EXPECT_FALSE(ref::band(theta - 1, 100, theta, 0, {0, 1}, 5, 100));
        EXPECT_TRUE(
            ref::band(-theta, 0, theta, 0, {0, static_cast<std::uint64_t>(2 * theta)}, 5, 100));
        EXPECT_FALSE(
            ref::band(-theta, 0, theta, 0, {0, static_cast<std::uint64_t>(theta - 1)}, 5, 100));

        for (unsigned gx = 0; gx < 5; ++gx) {
            for (unsigned gy = gx + 1; gy < 6; ++gy) {
                for (unsigned trial = 0; trial < 17; ++trial) {
                    std::int64_t contribution = 0;
                    const unsigned h = 4, Wt = 12;
                    const auto point_time =
                        (std::uint64_t(gx) << (Wt + h)) + ((trial * 7 + 1) << h) + 1;
                    for (unsigned r = 0; r < 2; ++r) {
                        for (unsigned negative = 0; negative < 2; ++negative) {
                            const auto query_time =
                                (std::uint64_t(gy) << (Wt + h)) + ((negative ? 3 : 33) << h) + 2;
                            const auto point_amount = ((trial * 31 + r) << h) + 1;
                            const auto query_amount = ((trial * 13 + 1) << h) + 2;
                            contribution +=
                                (point_time < query_time && point_amount < query_amount) *
                                (negative ? -1 : 1) * (r ? -1 : 1);
                        }
                    }
                    EXPECT_EQ(contribution, 0);
                }
            }
        }
    }

    std::vector<Host> local_hosts_v8(unsigned n) {
        std::vector<Host> hosts;
        std::vector<int> sockets;
        for (unsigned i = 0; i < n; ++i) {
            const auto fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                throw std::runtime_error("socket");
            }
            sockets.push_back(fd);
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a))) {
                throw std::runtime_error("bind");
            }
            socklen_t length = sizeof(a);
            ::getsockname(fd, reinterpret_cast<sockaddr *>(&a), &length);
            hosts.push_back({"127.0.0.1", ntohs(a.sin_port)});
        }
        for (auto fd : sockets) {
            ::close(fd);
        }
        return hosts;
    }

    TEST(V8, AllPatternsUseIdenticalCompressedSlotObjects) {
        for (auto pattern : {"P1", "P2", "P3", "P4"}) {
            Temp tmp;
            InvestigationConfig c;
            c.pattern = pattern;
            c.algorithm = "dc";
            c.n = 3;
            c.initiator = 2;
            c.rows = {6, 6, 6};
            c.kin = 4;
            c.kout = 3;
            c.windows = {{0, 6}};
            c.amount_lower = 99;
            c.amount_upper = 104;
            c.min_cluster = 2;
            auto p = compile_investigation(c);
            auto nc = c;
            nc.algorithm = "nl";
            auto q = compile_investigation(nc);

            for (auto m : q.masks) {
                if (m.name.starts_with("pair.") || m.name == "pattern.amount.mask") {
                    bool exists = false;
                    for (const auto &old : p.masks) {
                        exists |= old.name == m.name;
                    }
                    if (exists) {
                        continue;
                    }
                    for (auto &t : m.terms) {
                        const auto name = t.source < q.inputs.size()
                                              ? q.inputs[t.source].name
                                              : q.masks[t.source - q.inputs.size()].name;
                        t.source = named_mask(p, name);
                    }
                    p.masks.push_back(std::move(m));
                }
            }
            const auto cmp_old = p.comparisons.size(), gate_old = p.gates.size(),
                       bitmul_old = p.bitmuls.size();
            std::vector<std::string> cmp_names, gate_names, bitmul_names;
            for (auto v : q.comparisons) {
                if (!v.name.starts_with("compact.")) {
                    for (auto &id : v.inputs) {
                        const auto name = id < q.inputs.size() ? q.inputs[id].name
                                                               : q.masks[id - q.inputs.size()].name;
                        id = named_mask(p, name);
                    }
                    cmp_names.push_back(v.name);
                    v.name += ".nl";
                    p.comparisons.push_back(std::move(v));
                }
            }
            for (auto v : q.gates) {
                if (!v.name.starts_with("compact.")) {
                    gate_names.push_back(v.name);
                    v.name += ".nl";
                    p.gates.push_back(std::move(v));
                }
            }
            for (auto v : q.bitmuls) {
                const auto name = v.mask < q.inputs.size() ? q.inputs[v.mask].name
                                                           : q.masks[v.mask - q.inputs.size()].name;
                v.mask = named_mask(p, name);
                bitmul_names.push_back(v.name);
                v.name += ".nl";
                p.bitmuls.push_back(std::move(v));
            }
            p.validate();
            generate_materials(p, MaterialMode::Prg, tmp.path / "m");
            auto view = p;
            auto rename = [](auto &vec, std::size_t old, const auto &names) {
                for (std::size_t i = 0; i < old; ++i) {
                    if (std::find(names.begin(), names.end(), vec[i].name) != names.end()) {
                        vec[i].name = "unused.dc." + vec[i].name;
                    }
                }
                for (std::size_t i = old; i < vec.size(); ++i) {
                    vec[i].name.resize(vec[i].name.size() - 3);
                }
            };
            rename(view.comparisons, cmp_old, cmp_names);
            rename(view.gates, gate_old, gate_names);
            rename(view.bitmuls, bitmul_old, bitmul_names);
            std::array<std::unique_ptr<MaterialStore>, 3> ss;
            std::array<std::unique_ptr<InputManager>, 3> input;
            std::array<std::unique_ptr<Compact>, 3> compact;
            std::array<std::unique_ptr<DominanceCount>, 3> dc;
            std::array<std::unique_ptr<NestedLoop>, 3> nl;
            std::vector<ref::PlainRow> rows;
            for (unsigned party = 0; party < 3; ++party) {
                nlohmann::json j = {{"inputs", nlohmann::json::object()}};
                const auto suffix = "_" + std::to_string(party);
                for (auto name : {"sid", "dir", "t", "a"}) {
                    j["inputs"][std::string(name) + suffix] = nlohmann::json::array();
                }
                if (c.carry_cp()) {
                    j["inputs"]["cp" + suffix] = nlohmann::json::array();
                }
                for (unsigned i = 0; i < 6; ++i) {
                    ref::PlainRow row;
                    row.subject = 42;
                    row.outgoing = i % 2;
                    row.time = i / 2;
                    row.amount = 100 + i % 3;
                    row.counterparty = i % 3;
                    rows.push_back(row);
                    j["inputs"]["sid" + suffix].push_back(42);
                    j["inputs"]["dir" + suffix].push_back(row.outgoing);
                    j["inputs"]["t" + suffix].push_back(row.time);
                    j["inputs"]["a" + suffix].push_back(static_cast<std::uint64_t>(row.amount));
                    if (c.carry_cp()) {
                        j["inputs"]["cp" + suffix].push_back(row.counterparty);
                    }
                }
                if (party == 2) {
                    j["inputs"]["subject"] = {42};
                }
                const auto path = tmp.path / ("input" + std::to_string(party) + ".json");
                {
                    std::ofstream f(path);
                    f << j;
                }
                ss[party] = std::make_unique<MaterialStore>(
                    p, party, tmp.path / "m" / ("party_" + std::to_string(party) + ".bin"));
                ss[party]->preload();
                input[party] = std::make_unique<InputManager>(*ss[party], p, path);
                compact[party] = std::make_unique<Compact>(c, *ss[party], p, *input[party]);
                dc[party] = std::make_unique<DominanceCount>(c,
                                                             *ss[party],
                                                             p,
                                                             compact[party]->inbound(),
                                                             compact[party]->outbound(),
                                                             compact[party]->omega_in(),
                                                             compact[party]->omega_out());
                nl[party] = std::make_unique<NestedLoop>(nc,
                                                         *ss[party],
                                                         view,
                                                         compact[party]->inbound(),
                                                         compact[party]->outbound(),
                                                         compact[party]->omega_in(),
                                                         compact[party]->omega_out());
                ss[party]->claim_execution();
                ss[party]->begin_online();
            }
            Config config;
            config.hosts = local_hosts_v8(3);
            config.threads = 2;
            config.timeout_ms = 5000;
            std::array<std::exception_ptr, 3> errors{};
            std::vector<std::thread> threads;
            for (unsigned party = 0; party < 3; ++party) {
                threads.emplace_back([&, party] {
                    try {
                        Net net(party, config);
                        input[party]->run(net);
                        compact[party]->run(net);
                        dc[party]->run(net);
                        nl[party]->run(net);
                    } catch (...) {
                        errors[party] = std::current_exception();
                    }
                });
            }
            for (auto &t : threads) {
                t.join();
            }
            for (auto e : errors) {
                if (e) {
                    std::rethrow_exception(e);
                }
            }
            std::uint64_t d = 0, n = 0, od = 0, on = 0;
            for (unsigned party = 0; party < 3; ++party) {
                d += dc[party]->count_shares()[0];
                n += nl[party]->count_shares()[0];
                od += dc[party]->overflow_share();
                on += nl[party]->overflow_share();
            }
            const auto ideal = ref::investigation(rows, 42, c);
            EXPECT_EQ(d, ideal.counts[0]) << pattern;
            EXPECT_EQ(n, d) << pattern;
            EXPECT_EQ(od, on);
            EXPECT_EQ(od, ideal.overflow);
            EXPECT_GT(d, 0u);
        }
    }

}
