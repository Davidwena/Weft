#include "bj/net/net.hpp"
#include "bj/common/aligned.hpp"
#include "bj/common/ring.hpp"
#include "allocation.hpp"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <thread>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
    using namespace bj;

    Config make_config(unsigned n, OpenMode mode, unsigned king) {
        Config c;
        c.timeout_ms = 3000;
        c.threads = 1;
        c.open_mode = mode;
        c.king = king;
        std::vector<int> fds(n, -1);
        for (auto &fd : fds) {
            fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                throw std::runtime_error("test socket");
            }
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a))) {
                throw std::runtime_error("test bind");
            }
            socklen_t len = sizeof(a);
            ::getsockname(fd, reinterpret_cast<sockaddr *>(&a), &len);
            c.hosts.push_back({"127.0.0.1", ntohs(a.sin_port)});
        }
        for (auto fd : fds) {
            ::close(fd);
        }
        return c;
    }

    template <unsigned K> void fill_ring(std::span<std::byte> bytes, unsigned id) {
        using T = typename Ring<K>::T;

        std::vector<T> values(50003);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] =
                Ring<K>::normalize(static_cast<T>((UINT64_C(0x9e3779b97f4a7c15) * (i + 1)) ^
                                                  (UINT64_C(0xffffffffffffffff) - id * 97)));
        }
        pack_ring<K>(values, bytes);
    }

    template <unsigned K> bool verify_ring(std::span<const std::byte> bytes, unsigned n) {
        using T = typename Ring<K>::T;
        std::vector<T> values(50003);
        unpack_ring<K>(bytes, values);
        for (std::size_t i = 0; i < values.size(); ++i) {
            T expected = 0;
            for (unsigned id = 0; id < n; ++id) {
                expected = Ring<K>::add(
                    expected,
                    Ring<K>::normalize(static_cast<T>((UINT64_C(0x9e3779b97f4a7c15) * (i + 1)) ^
                                                      (UINT64_C(0xffffffffffffffff) - id * 97))));
            }
            if (values[i] != expected) {
                return false;
            }
        }
        return true;
    }

    void run_open(unsigned n, OpenMode mode, unsigned king) {
        auto c = make_config(n, mode, king);
        OpenLayout layout;
        layout.add_ring(22, 50003)
            .add_ring(32, 50003)
            .add_ring(49, 50003)
            .add_ring(64, 50003)
            .add_bits(524291);
        layout.seal();
        std::barrier ready(static_cast<std::ptrdiff_t>(n + 1)),
            go(static_cast<std::ptrdiff_t>(n + 1)), end(static_cast<std::ptrdiff_t>(n + 1)),
            release(static_cast<std::ptrdiff_t>(n + 1));
        std::vector<std::thread> parties;
        std::vector<std::exception_ptr> errors(n);
        std::vector<unsigned> correct(n, 0);
        std::vector<OpenStats> counters(n);
        std::vector<std::vector<PeerStats>> traffic(n);
        for (auto &t : traffic) {
            t.resize(n);
        }
        for (unsigned id = 0; id < n; ++id) {
            parties.emplace_back([&, id] {
                bool at_ready = false, at_end = false;
                try {
                    Net net(id, c);
                    AlignedBuffer<std::byte> in(layout.bytes()), out(layout.bytes());
                    OpenBuf buf(layout, in.span(), out.span());
                    auto fields = layout.fields();
                    fill_ring<22>(in.span().subspan(fields[0].offset, fields[0].bytes), id);
                    fill_ring<32>(in.span().subspan(fields[1].offset, fields[1].bytes), id);
                    fill_ring<49>(in.span().subspan(fields[2].offset, fields[2].bytes), id);
                    fill_ring<64>(in.span().subspan(fields[3].offset, fields[3].bytes), id);
                    auto bits = in.span().subspan(fields[4].offset, fields[4].bytes);
                    for (std::size_t i = 0; i < bits.size(); ++i) {
                        bits[i] = std::byte((i * 31 + id * 17) & 255);
                    }
                    net.open_round(buf);
                    net.reset_stats();
                    at_ready = true;
                    ready.arrive_and_wait();
                    go.arrive_and_wait();
                    for (unsigned repeat = 0; repeat < 3; ++repeat) {
                        net.open_round(buf);
                    }
                    at_end = true;
                    end.arrive_and_wait();
                    release.arrive_and_wait();
                    counters[id] = net.open_stats();
                    std::copy(net.stats().begin(), net.stats().end(), traffic[id].begin());
                    bool ok =
                        verify_ring<22>(out.span().subspan(fields[0].offset, fields[0].bytes), n) &&
                        verify_ring<32>(out.span().subspan(fields[1].offset, fields[1].bytes), n) &&
                        verify_ring<49>(out.span().subspan(fields[2].offset, fields[2].bytes), n) &&
                        verify_ring<64>(out.span().subspan(fields[3].offset, fields[3].bytes), n);
                    auto opened_bits = out.span().subspan(fields[4].offset, fields[4].bytes);
                    for (std::size_t i = 0; i < opened_bits.size(); ++i) {
                        unsigned expected = 0;
                        for (unsigned p = 0; p < n; ++p) {
                            expected ^= (i * 31 + p * 17) & 255;
                        }
                        if (i + 1 == opened_bits.size()) {
                            expected &= 7;
                        }
                        ok &= (opened_bits[i] == std::byte(expected));
                    }
                    ok &= net.receive_scratch_bytes() == (n - 1) * OpenLayout::chunk_bytes;
                    correct[id] = ok;
                } catch (...) {
                    errors[id] = std::current_exception();
                    if (!at_ready) {
                        ready.arrive_and_wait();
                        go.arrive_and_wait();
                    }
                    if (!at_end) {
                        end.arrive_and_wait();
                        release.arrive_and_wait();
                    }
                }
            });
        }
        ready.arrive_and_wait();
        test::allocations = 0;
        test::count_allocations = true;
        go.arrive_and_wait();
        end.arrive_and_wait();
        test::count_allocations = false;
        const auto allocs = test::allocations.load();
        release.arrive_and_wait();
        for (auto &t : parties) {
            t.join();
        }
        for (const auto &e : errors) {
            if (e) {
                std::rethrow_exception(e);
            }
        }
        EXPECT_EQ(allocs, 0);
        const bool star = n >= 3 && mode == OpenMode::Auto;
        for (unsigned id = 0; id < n; ++id) {
            EXPECT_TRUE(correct[id]);
            EXPECT_EQ(counters[id].rounds, 3);
            EXPECT_EQ(counters[id].hops, star ? 6 : 3);
            EXPECT_EQ(counters[id].contribution_bits, 3 * layout.bits());
            if (!star || id == king) {
                EXPECT_GT(counters[id].chunks_before_all_received, 0);
            } else {
                EXPECT_EQ(counters[id].reduced_chunks, 0);
            }
            for (unsigned p = 0; p < n; ++p) {
                const auto messages = (p != id && (!star || id == king || p == king)) ? 3u : 0u;
                const auto &s = traffic[id][p];
                EXPECT_EQ(s.sent_messages, messages);
                EXPECT_EQ(s.received_messages, messages);
                EXPECT_EQ(s.sent_payload, messages * layout.bytes());
                EXPECT_EQ(s.received_payload, s.sent_payload);
                EXPECT_EQ(s.sent_wire, messages * (layout.bytes() + 16));
                EXPECT_EQ(s.received_wire, s.sent_wire);
            }
        }
    }

    TEST(Open, TwoPartyAuto) {
        run_open(2, OpenMode::Auto, 0);
    }

    TEST(Open, ThreePartyKing) {
        run_open(3, OpenMode::Auto, 0);
    }

    TEST(Open, FivePartyRotatedKing) {
        run_open(5, OpenMode::Auto, 3);
    }

    TEST(Open, ThreePartyBroadcast) {
        run_open(3, OpenMode::Broadcast, 0);
    }

    TEST(Open, FivePartyBroadcast) {
        run_open(5, OpenMode::Broadcast, 2);
    }

    TEST(Open, NonOpenRoutesUnaffected) {
        auto c = make_config(3, OpenMode::Auto, 0);
        std::array<std::exception_ptr, 3> errors;
        std::array<std::vector<PeerStats>, 3> traffic;
        std::array<std::thread, 3> parties;
        std::array<bool, 3> correct{};
        for (unsigned id = 0; id < 3; ++id) {
            parties[id] = std::thread([&, id] {
                try {
                    Net net(id, c);
                    std::array<std::byte, 7> buf{};

                    if (id == 1) {
                        buf.fill(std::byte{0x31});
                        net.send(0, buf);
                        net.send(2, buf);
                    } else {
                        net.recv(1, buf);
                    }

                    if (id == 0) {
                        net.send(1, buf);
                    } else {
                        net.recv(id - 1, buf);
                        if (id == 1) {
                            net.send(2, buf);
                        }
                    }
                    if (id == 2) {
                        net.send(0, buf);
                        net.send(1, buf);
                    } else {
                        net.recv(2, buf);
                    }

                    if (id == 2) {
                        net.recv(0, buf);
                        net.recv(1, buf);
                    } else {
                        net.send(2, buf);
                    }
                    traffic[id] = net.stats();
                    correct[id] = std::all_of(buf.begin(),
                                              buf.end(),
                                              [](auto v) {
                                                  return v == std::byte{0x31};
                                              }) &&
                                  net.open_stats().rounds == 0 && net.open_stats().hops == 0;
                } catch (...) {
                    errors[id] = std::current_exception();
                }
            });
        }
        for (auto &t : parties) {
            t.join();
        }
        for (const auto &e : errors) {
            if (e) {
                std::rethrow_exception(e);
            }
        }
        constexpr unsigned messages[3][3] = {{0, 1, 1}, {1, 0, 3}, {1, 1, 0}};
        for (unsigned id = 0; id < 3; ++id) {
            EXPECT_TRUE(correct[id]);
            for (unsigned p = 0; p < 3; ++p) {
                EXPECT_EQ(traffic[id][p].sent_messages, messages[id][p]);
                EXPECT_EQ(traffic[id][p].received_messages, messages[p][id]);
                EXPECT_EQ(traffic[id][p].sent_payload, 7 * messages[id][p]);
                EXPECT_EQ(traffic[id][p].sent_wire, 15 * messages[id][p]);
            }
        }
    }

    TEST(Open, SchemaValidationAndChunks) {
        OpenLayout l;
        l.add_ring(22, 50003).add_bits(17);
        l.seal();
        EXPECT_GT(l.chunks().size(), 2);
        for (const auto &c : l.chunks()) {
            EXPECT_LE(c.bytes, OpenLayout::chunk_bytes);
        }
        EXPECT_THROW(l.add_bits(1), std::logic_error);
        AlignedBuffer<std::byte> a(l.bytes()), b(l.bytes());
        EXPECT_THROW((OpenBuf{l, a.span(), a.span()}), std::invalid_argument);
        EXPECT_THROW((OpenBuf{l, a.span().first(1), b.span()}), std::invalid_argument);
        OpenLayout bad;
        EXPECT_THROW(bad.add_ring(0, 1), std::invalid_argument);
        EXPECT_THROW(bad.add_ring(97, 1), std::invalid_argument);
        EXPECT_THROW(bad.add_ring(64, SIZE_MAX), std::length_error);
    }

    TEST(Open, LayoutFingerprintChecked) {
        auto c = make_config(2, OpenMode::Auto, 0);
        std::array<std::exception_ptr, 2> errors;
        std::array<std::thread, 2> parties;
        for (unsigned id = 0; id < 2; ++id) {
            parties[id] = std::thread([&, id] {
                try {
                    Net net(id, c);
                    OpenLayout l;
                    if (id) {
                        l.add_ring(32, 2);
                    } else {
                        l.add_ring(64, 1);
                    }
                    l.seal();
                    AlignedBuffer<std::byte> in(8), out(8);
                    std::fill(in.span().begin(), in.span().end(), std::byte{0});
                    OpenBuf buf(l, in.span(), out.span());
                    net.open_round(buf);
                } catch (...) {
                    errors[id] = std::current_exception();
                }
            });
        }
        for (auto &t : parties) {
            t.join();
        }
        EXPECT_TRUE(errors[0]);
        EXPECT_TRUE(errors[1]);
    }
}
