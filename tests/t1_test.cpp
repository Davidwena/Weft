#include "bj/core/material.hpp"
#include "bj/core/input.hpp"
#include "bj/core/share.hpp"
#include "bj/core/round_exec.hpp"
#include "bj/proto/b2a.hpp"
#include "bj/proto/and2a.hpp"
#include "allocation.hpp"
#include "bj/ref/plaintext.hpp"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <barrier>
#include <cstring>
#include <fstream>
#include <thread>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <omp.h>

namespace {
    using namespace bj;

    struct Temp {
        std::filesystem::path path;

        Temp() {
            char pattern[] = "/tmp/bj-t1-XXXXXX";
            auto *p = ::mkdtemp(pattern);
            if (!p) {
                throw std::runtime_error("mkdtemp");
            }
            path = p;
        }

        ~Temp() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    };

    MaterialPlan plan_for(unsigned n, std::size_t count, bool uniform = false) {
        MaterialPlan p;
        p.n = n;
        p.king = n - 1;
        p.output_party = n - 1;
        for (unsigned j = 0; j < 4; ++j) {
            p.inputs.push_back({"b" + std::to_string(j), (j + 1) % n, 0, count});
        }
        p.inputs.push_back({"amount", 0, 22, count});
        p.inputs.push_back({"wide", 1, 64, count});
        constexpr unsigned widths[] = {7, 22, 49, 64};
        for (unsigned m = 1; m <= 4; ++m) {
            GateSpec g;
            g.name = "g" + std::to_string(m);
            g.m = m;
            g.width = uniform ? 32 : widths[m - 1];
            g.count = count;
            for (unsigned j = 0; j < m; ++j) {
                g.inputs.push_back(j);
            }
            p.gates.push_back(g);
        }
        p.validate();
        return p;
    }

    std::filesystem::path mat_file(const std::filesystem::path &d, unsigned id) {
        return d / ("party_" + std::to_string(id) + ".bin");
    }

    TEST(AllocationCounter, AllNewForms) {
        test::allocations = 0;
        test::count_allocations = true;
        void *a = ::operator new(16);
        void *b = ::operator new[](16);
        void *c = ::operator new(16, std::nothrow);
        void *d = ::operator new[](16, std::nothrow);
        void *e = ::operator new(64, std::align_val_t(64));
        void *f = ::operator new[](64, std::align_val_t(64));
        void *g = ::operator new(64, std::align_val_t(64), std::nothrow);
        void *h = ::operator new[](64, std::align_val_t(64), std::nothrow);
        test::count_allocations = false;
        const auto count = test::allocations.load();
        ::operator delete(a);
        ::operator delete[](b);
        ::operator delete(c, std::nothrow);
        ::operator delete[](d, std::nothrow);
        ::operator delete(e, std::align_val_t(64));
        ::operator delete[](f, std::align_val_t(64));
        ::operator delete(g, std::align_val_t(64), std::nothrow);
        ::operator delete[](h, std::align_val_t(64), std::nothrow);
        EXPECT_EQ(count, 8);
    }

    TEST(Packed, ArbitraryWidthsAndOffsets) {
        for (unsigned k = 1; k <= 64; ++k) {
            for (unsigned offset = 0; offset < 64; ++offset) {
                std::array<std::byte, 40> data{};
                data.fill(std::byte{0xa5});
                const auto original = data;
                const auto value = UINT64_C(0xfedcba9876543210) & width_mask(k);
                store_packed(data, offset, k, value);
                ASSERT_EQ(load_packed(data, offset, k), value);
                for (unsigned bit = 0; bit < data.size() * 8; ++bit) {
                    if (bit < offset || bit >= offset + k) {
                        ASSERT_EQ(load_packed(data, bit, 1), load_packed(original, bit, 1));
                    }
                }
            }
        }
        for (std::size_t offset = 0; offset < 64; ++offset) {
            for (std::size_t count : {0, 1, 7, 8, 63, 64, 65, 137}) {
                std::array<std::byte, 32> wire{};
                std::vector<std::uint64_t> in(bits_words(count), UINT64_C(0xfedcba9876543210)),
                    out(in.size());
                if (count % 64) {
                    in.back() &= width_mask(static_cast<unsigned>(count % 64));
                }
                store_bits(wire, offset, count, in);
                load_bits(wire, offset, count, out);
                ASSERT_EQ(in, out);
            }
        }
    }

    template <unsigned K> void sharing_algebra() {
        using T = typename Ring<K>::T;
        for (unsigned n : {2u, 3u, 5u}) {
            std::vector<std::array<T, 19>> a(n), b(n), out(n), delta(n), lambda(n);
            for (unsigned id = 0; id < n; ++id) {
                for (unsigned i = 0; i < 19; ++i) {
                    a[id][i] = Ring<K>::normalize(static_cast<T>(UINT64_MAX - id * 91 - i * 17));
                    b[id][i] =
                        Ring<K>::normalize(static_cast<T>(UINT64_C(0x123456789abcdef0) + id + i));
                    lambda[id][i] = b[id][i];
                    delta[id][i] = static_cast<T>(i + 7);
                }
            }
            for (unsigned id = 0; id < n; ++id) {
                share_add_public<K>(a[id], T{13}, id, out[id]);
            }
            for (unsigned i = 0; i < 19; ++i) {
                T expected = 13, actual = 0;
                for (unsigned id = 0; id < n; ++id) {
                    expected = Ring<K>::add(expected, a[id][i]);
                    actual = Ring<K>::add(actual, out[id][i]);
                }
                ASSERT_EQ(actual, expected);
            }
            for (unsigned id = 0; id < n; ++id) {
                share_add<K>(a[id], b[id], out[id]);
                share_sub<K>(out[id], b[id], out[id]);
                EXPECT_EQ(out[id], a[id]);
                share_scale<K>(a[id], T{3}, out[id]);
            }
            for (unsigned i = 0; i < 19; ++i) {
                T expected = 0, actual = 0;
                for (unsigned id = 0; id < n; ++id) {
                    expected = Ring<K>::add(expected, a[id][i]);
                    actual = Ring<K>::add(actual, out[id][i]);
                }
                ASSERT_EQ(actual, Ring<K>::mul(expected, 3));
            }
            for (unsigned id = 0; id < n; ++id) {
                masked_to_additive<K>({delta[id], lambda[id]}, id, out[id]);
            }
            for (unsigned i = 0; i < 19; ++i) {
                T actual = 0, mask = 0;
                for (unsigned id = 0; id < n; ++id) {
                    actual = Ring<K>::add(actual, out[id][i]);
                    mask = Ring<K>::add(mask, lambda[id][i]);
                }
                ASSERT_EQ(actual, Ring<K>::sub(delta[0][i], mask));
            }
        }
    }

    TEST(Share, NativeAlgebraAndMaskedConversion) {
        sharing_algebra<7>();
        sharing_algebra<22>();
        sharing_algebra<32>();
        sharing_algebra<49>();
        sharing_algebra<64>();
    }

    TEST(Share, LocalReductionAndXorConstants) {
        std::array<std::uint64_t, 4> a = {UINT64_MAX, 0, 1, UINT64_C(1) << 63};
        std::array<std::uint32_t, 4> out{};
        share_reduce<64, 22>(a, out);
        for (unsigned i = 0; i < 4; ++i) {
            EXPECT_EQ(out[i], a[i] & Ring<22>::mask());
        }
        std::array<std::uint64_t, 2> x = {0xaaaaaaaaaaaaaaaa, 0x123456789abcdef0},
                                     c = {0xffffffffffffffff, 0x1111111111111111}, y{}, z{};
        xor_public(x, c, 0, y);
        xor_public(x, c, 1, z);
        EXPECT_EQ(y[0] ^ z[0], c[0]);
        xor_shares(y, z, y);
        EXPECT_EQ(y, c);
    }

    TEST(Material, CorrelationsAndPerPartyLayout) {
        for (auto mode : {MaterialMode::Random, MaterialMode::Prg}) {
            for (unsigned n : {2u, 3u, 5u}) {
                Temp temp;
                const auto package = temp.path / "material";
                const auto p = plan_for(n, 73);
                const auto report = generate_materials(p, mode, package);
                std::vector<std::unique_ptr<MaterialStore>> stores;
                for (unsigned id = 0; id < n; ++id) {
                    stores.push_back(std::make_unique<MaterialStore>(p, id, mat_file(package, id)));
                    stores.back()->preload();
                }
                const auto &layout = stores[0]->layout();
                std::uint64_t expected_bits = 0;
                for (std::size_t g = 0; g < p.gates.size(); ++g) {
                    const auto &gate = p.gates[g];
                    expected_bits += gate.count * (gate.m + ((1u << gate.m) - 1) * gate.width);
                    for (std::size_t i = 0; i < gate.count; ++i) {
                        unsigned tau[4]{};
                        for (unsigned j = 0; j < gate.m; ++j) {
                            for (unsigned id = 0; id < n; ++id) {
                                tau[j] ^= static_cast<unsigned>(
                                    load_packed(stores[id]->packed(layout.gate_masks[g]),
                                                j * gate.count + i,
                                                1));
                            }
                        }
                        for (unsigned subset = 1; subset < (1u << gate.m); ++subset) {
                            std::uint64_t sum = 0;
                            unsigned product = 1;
                            for (unsigned j = 0; j < gate.m; ++j) {
                                if (subset & (1u << j)) {
                                    product &= tau[j];
                                }
                            }
                            for (unsigned id = 0; id < n; ++id) {
                                sum += load_packed(
                                    stores[id]->packed(layout.gate_products[g][subset - 1]),
                                    i * gate.width,
                                    gate.width);
                            }
                            ASSERT_EQ(sum & width_mask(gate.width), product);
                        }
                    }
                }
                for (std::size_t index = 0; index < p.inputs.size(); ++index) {
                    const auto &in = p.inputs[index];
                    const auto width = in.width ? in.width : 1;
                    const auto actual = stores[in.owner]->packed(layout.owner_masks[index]);
                    for (unsigned id = 0; id < n; ++id) {
                        if (id != in.owner) {
                            EXPECT_THROW(stores[id]->packed(layout.owner_masks[index]),
                                         std::logic_error);
                        }
                    }
                    for (std::size_t i = 0; i < in.count; ++i) {
                        std::uint64_t value = 0;
                        for (unsigned id = 0; id < n; ++id) {
                            const auto x = load_packed(
                                stores[id]->packed(layout.input_shares[index]), i * width, width);
                            if (in.width) {
                                value += x;
                            } else {
                                value ^= x;
                            }
                        }
                        ASSERT_EQ(value & width_mask(width), load_packed(actual, i * width, width));
                    }
                }
                for (unsigned id = 0; id < n; ++id) {
                    const auto &l = stores[id]->layout();
                    EXPECT_EQ(l.gate_bits, expected_bits);
                    EXPECT_EQ(report.gate_bits[id], expected_bits);
                    EXPECT_EQ(std::filesystem::file_size(mat_file(package, id)), l.file_bytes);
                    EXPECT_EQ(l.seed_bytes, mode == MaterialMode::Prg && id + 1 < n ? 16 : 0);
                    for (const auto &seg : l.segments) {
                        if (seg.role != SegmentRole::OwnerMask) {
                            EXPECT_EQ(seg.storage,
                                      mode == MaterialMode::Prg && id + 1 < n
                                          ? SegmentStorage::Seeded
                                          : SegmentStorage::Dense);
                        }
                    }
                }
                EXPECT_THROW(generate_materials(p, mode, package), std::runtime_error);
            }
        }
    }

    TEST(Material, ValidationAndOneTimeGuards) {
        Temp t;
        const auto p = plan_for(3, 17);
        const auto package = t.path / "material";
        generate_materials(p, MaterialMode::Prg, package);
        MaterialStore s(p, 0, mat_file(package, 0));
        EXPECT_THROW(s.packed(0), std::logic_error);
        EXPECT_THROW(s.begin_online(), std::logic_error);
        s.preload();
        s.claim_input(0);
        s.claim_gate(0);
        EXPECT_THROW(s.claim_input(0), std::logic_error);
        EXPECT_THROW(s.claim_gate(0), std::logic_error);
        s.claim_execution();
        EXPECT_THROW(s.claim_execution(), std::logic_error);
        MaterialStore second(p, 0, mat_file(package, 0));
        EXPECT_THROW(second.claim_execution(), std::runtime_error);
        s.begin_online();
        EXPECT_THROW(s.packed(0), std::logic_error);
        EXPECT_THROW(s.preload(), std::logic_error);
        EXPECT_THROW((MaterialStore(p, 1, mat_file(package, 0))), std::runtime_error);
        auto wrong = p;
        wrong.gates[0].width = 8;
        EXPECT_THROW((MaterialStore(wrong, 0, mat_file(package, 0))), std::runtime_error);
        const auto damaged = package / "bad.bin";
        std::filesystem::copy_file(mat_file(package, 0), damaged);
        {
            std::fstream file(damaged, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(256 + 24);
            file.put(char(7));
        }
        EXPECT_THROW((MaterialStore(p, 0, damaged)), std::runtime_error);
        std::filesystem::resize_file(damaged, 10);
        EXPECT_THROW((MaterialStore(p, 0, damaged)), std::runtime_error);
        std::filesystem::remove(package / "READY");
        EXPECT_THROW((MaterialStore(p, 0, mat_file(package, 0))), std::runtime_error);
    }

    TEST(RoundExecutor, MixedSchemaAndUnalignedBitSlots) {
        RoundPlan p;
        const auto r = p.ring(22, 13);
        const auto a = p.bits(73), b = p.bits(17), c = p.bits(129);
        p.seal();
        std::vector<std::byte> data(p.layout().bytes(), std::byte{0});
        OpenWriter writer(p, data);
        OpenReader reader(p, data);
        std::array<std::uint32_t, 13> ring{}, read{};
        for (unsigned i = 0; i < 13; ++i) {
            ring[i] = i * 113;
        }
        std::array<std::uint64_t, 2> aa = {UINT64_MAX, 0x1ff}, aaa{};
        std::array<std::uint64_t, 1> bb = {0x11111}, bbb{};
        std::array<std::uint64_t, 3> cc = {0x1234, UINT64_MAX, 1}, ccc{};
        writer.ring<std::uint32_t>(r, ring);
        writer.bits(a, aa);
        writer.bits(b, bb);
        writer.bits(c, cc);
        reader.ring<std::uint32_t>(r, read);
        reader.bits(a, aaa);
        reader.bits(b, bbb);
        reader.bits(c, ccc);
        EXPECT_EQ(ring, read);
        EXPECT_EQ(aa, aaa);
        EXPECT_EQ(bb, bbb);
        EXPECT_EQ(cc, ccc);
        EXPECT_EQ(p.layout().bits(), 13 * 22 + 73 + 17 + 129);
    }

    Config network_config(unsigned n) {
        Config c;
        c.threads = 2;
        c.timeout_ms = 10000;
        c.king = 1;
        std::vector<int> fds;
        for (unsigned id = 0; id < n; ++id) {
            int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                throw std::runtime_error("test socket");
            }
            fds.push_back(fd);
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a))) {
                throw std::runtime_error("test bind");
            }
            socklen_t size = sizeof(a);
            ::getsockname(fd, reinterpret_cast<sockaddr *>(&a), &size);
            c.hosts.push_back({"127.0.0.1", ntohs(a.sin_port)});
        }
        for (int fd : fds) {
            ::close(fd);
        }
        return c;
    }

    TEST(Protocols, NoOnlineAllocationsMergedRoundAndPrivateOutput) {
        constexpr unsigned n = 3;
        constexpr std::size_t count = 16417;
        Temp t;
        auto p = plan_for(n, count, true);
        p.king = 1;
        p.output_party = 2;
        const auto package = t.path / "material";
        generate_materials(p, MaterialMode::Prg, package);
        for (unsigned id = 0; id < n; ++id) {
            nlohmann::json values = nlohmann::json::object();
            for (std::size_t j = 0; j < p.inputs.size(); ++j) {
                if (p.inputs[j].owner == id) {
                    auto a = nlohmann::json::array();
                    for (std::size_t i = 0; i < count; ++i) {
                        a.push_back(j < 4 ? ((i >> j) & 1) : (i * 101 + 3));
                    }
                    values[p.inputs[j].name] = a;
                }
            }
            std::ofstream f(t.path / ("input_" + std::to_string(id) + ".json"));
            f << nlohmann::json{{"inputs", values}};
        }
        const auto cfg = network_config(n);
        std::barrier ready(4), go(4), done(4), release(4);
        std::array<std::exception_ptr, n> errors;
        std::array<unsigned, n> ok{};
        std::array<OpenStats, n> opens;
        std::array<std::vector<std::uint32_t>, n> amounts;
        for (auto &col : amounts) {
            col.resize(count);
        }
        std::array<std::thread, n> parties;
        for (unsigned id = 0; id < n; ++id) {
            parties[id] = std::thread([&, id] {
                bool at_ready = false, at_done = false;
                try {
                    omp_set_num_threads(2);
                    MaterialStore store(p, id, mat_file(package, id));
                    store.preload();
                    InputManager input(
                        store, p, t.path / ("input_" + std::to_string(id) + ".json"));
                    std::array<AlignedBuffer<std::uint32_t>, 4> out;
                    for (auto &col : out) {
                        col = AlignedBuffer<std::uint32_t>(count);
                    }
                    std::array<std::span<const std::uint64_t>, 1> v1 = {input.bits(0)};
                    std::array<std::span<const std::uint64_t>, 2> v2 = {input.bits(0),
                                                                        input.bits(1)};
                    std::array<std::span<const std::uint64_t>, 3> v3 = {
                        input.bits(0), input.bits(1), input.bits(2)};
                    std::array<std::span<const std::uint64_t>, 4> v4 = {
                        input.bits(0), input.bits(1), input.bits(2), input.bits(3)};
                    B2ABatch<32> b1(store, p, 0, v1, out[0].span());
                    And2ABatch<2, 32> b2(store, p, 1, v2, out[1].span());
                    And2ABatch<3, 32> b3(store, p, 2, v3, out[2].span());
                    And2ABatch<4, 32> b4(store, p, 3, v4, out[3].span());
                    std::array<Step *, 4> steps = {&b1, &b2, &b3, &b4};
                    RoundExecutor exec(steps);
                    OpenLayout output_layout;
                    for (unsigned j = 0; j < 4; ++j) {
                        output_layout.add_ring(32, count);
                    }
                    output_layout.seal();
                    AlignedBuffer<std::byte> share(output_layout.bytes()),
                        value(output_layout.bytes());
                    std::fill(value.span().begin(), value.span().end(), std::byte{0x7f});
                    OpenBuf output(output_layout, share.span(), value.span());
                    store.claim_execution();
                    Net net(id, cfg);
                    store.begin_online();
                    at_ready = true;
                    ready.arrive_and_wait();
                    go.arrive_and_wait();
                    input.run(net);
                    net.reset_stats();
                    exec.run(net);
                    opens[id] = net.open_stats();
                    for (unsigned j = 0; j < 4; ++j) {
                        const auto &f = output_layout.fields()[j];
                        pack_native<std::uint32_t>(
                            32, out[j].span(), share.span().subspan(f.offset, f.bytes));
                    }
                    net.reveal_to(2, output);
                    at_done = true;
                    done.arrive_and_wait();
                    release.arrive_and_wait();
                    std::copy(input.at(4).share32.span().begin(),
                              input.at(4).share32.span().end(),
                              amounts[id].begin());
                    bool valid = true;
                    if (id != 2) {
                        valid = std::all_of(value.span().begin(), value.span().end(), [](auto b) {
                            return b == std::byte{0x7f};
                        });
                    } else {
                        for (unsigned j = 0; j < 4; ++j) {
                            std::vector<std::uint32_t> actual(count);
                            const auto &f = output_layout.fields()[j];
                            unpack_native<std::uint32_t>(
                                32, value.span().subspan(f.offset, f.bytes), actual);
                            for (std::size_t i = 0; i < count; ++i) {
                                std::array<std::uint8_t, 4> plain{};
                                for (unsigned b = 0; b <= j; ++b) {
                                    plain[b] = (i >> b) & 1;
                                }
                                if (actual[i] !=
                                    ref::and2a(std::span<const std::uint8_t>(plain).first(j + 1))) {
                                    valid = false;
                                }
                            }
                        }
                    }
                    try {
                        exec.run(net);
                        valid = false;
                    } catch (const std::logic_error &) {
                    }
                    try {
                        input.run(net);
                        valid = false;
                    } catch (const std::logic_error &) {
                    }
                    ok[id] = valid;
                } catch (...) {
                    errors[id] = std::current_exception();
                    if (!at_ready) {
                        ready.arrive_and_wait();
                        go.arrive_and_wait();
                    }
                    if (!at_done) {
                        done.arrive_and_wait();
                        release.arrive_and_wait();
                    }
                }
            });
        }
        ready.arrive_and_wait();
        test::allocations = 0;
        test::count_allocations = true;
        go.arrive_and_wait();
        done.arrive_and_wait();
        test::count_allocations = false;
        const auto allocs = test::allocations.load();
        release.arrive_and_wait();
        for (auto &thread : parties) {
            thread.join();
        }
        for (const auto &e : errors) {
            if (e) {
                std::rethrow_exception(e);
            }
        }
        EXPECT_EQ(allocs, 0);
        for (unsigned id = 0; id < n; ++id) {
            EXPECT_TRUE(ok[id]);
            EXPECT_EQ(opens[id].rounds, 1);
            EXPECT_EQ(opens[id].hops, 2);
            EXPECT_EQ(opens[id].contribution_bits, 10 * count);
        }
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t x = 0;
            for (auto &col : amounts) {
                x += col[i];
            }
            EXPECT_EQ(x & Ring<22>::mask(), (i * 101 + 3) & Ring<22>::mask());
        }
    }
}
