#include "bj/net/net.hpp"
#include "bj/common/aligned.hpp"
#include "allocation.hpp"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

namespace {
    using namespace bj;

    Config ports() {
        Config c;
        c.timeout_ms = 3000;
        c.threads = 1;
        std::array<int, 2> fds{};
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

    TEST(Net, FramingCountersAndNoAllocations) {
        const auto c = ports();
        std::exception_ptr error;
        std::atomic<bool> begin{false}, prepared{false}, complete{false}, release{false};
        std::thread peer([&] {
            try {
                Net net(1, c);
                AlignedBuffer<std::byte> in(1 << 20), out(1 << 20);
                std::fill(out.span().begin(), out.span().end(), std::byte{2});
                std::array<std::span<const std::byte>, 2> tx{out.span(),
                                                             std::span<const std::byte>{}};
                std::array<std::span<std::byte>, 2> rx{in.span(), std::span<std::byte>{}};
                net.exchange_all(tx, rx);
                prepared = true;
                while (!begin.load()) {
                    std::this_thread::yield();
                }
                for (unsigned i = 0; i < 10; ++i) {
                    net.exchange_all(tx, rx);
                }
                complete = true;
                while (!release.load()) {
                    std::this_thread::yield();
                }
                for (auto v : in.span()) {
                    if (v != std::byte{1}) {
                        throw std::runtime_error("data mismatch");
                    }
                }
                std::array<std::byte, 7> ping{};
                net.recv(0, ping);
                net.send(0, ping);
                tx[0] = {};
                rx[0] = {};
                net.exchange_all(tx, rx);
            } catch (...) {
                error = std::current_exception();
                prepared = true;
                complete = true;
            }
        });
        try {
            Net net(0, c);
            AlignedBuffer<std::byte> in(1 << 20), out(1 << 20);
            std::fill(out.span().begin(), out.span().end(), std::byte{1});
            std::array<std::span<const std::byte>, 2> tx{std::span<const std::byte>{}, out.span()};
            std::array<std::span<std::byte>, 2> rx{std::span<std::byte>{}, in.span()};
            net.exchange_all(tx, rx);
            while (!prepared.load()) {
                std::this_thread::yield();
            }
            test::allocations = 0;
            test::count_allocations = true;
            begin = true;
            for (unsigned i = 0; i < 10; ++i) {
                net.exchange_all(tx, rx);
            }
            while (!complete.load()) {
                std::this_thread::yield();
            }
            test::count_allocations = false;
            release = true;
            EXPECT_EQ(test::allocations.load(), 0);
            EXPECT_TRUE(std::all_of(in.span().begin(), in.span().end(), [](auto v) {
                return v == std::byte{2};
            }));
            std::array<std::byte, 7> ping{}, pong{};
            net.send(1, ping);
            net.recv(1, pong);
            EXPECT_EQ(ping, pong);
            tx[1] = {};
            rx[1] = {};
            net.exchange_all(tx, rx);
            EXPECT_EQ(net.rounds(), 12);
            const auto &s = net.stats()[1];
            EXPECT_EQ(s.sent_payload, 11 * (1 << 20) + 7);
            EXPECT_EQ(s.received_payload, s.sent_payload);
            EXPECT_EQ(s.sent_messages, 13);
            EXPECT_EQ(s.received_messages, 13);
            EXPECT_EQ(s.sent_wire, s.sent_payload + 13 * 8);
            EXPECT_EQ(s.received_wire, s.sent_wire);
            EXPECT_EQ(net.stats()[0].sent_wire, 0);
            net.reset_stats();
            EXPECT_EQ(net.rounds(), 0);
            EXPECT_EQ(net.stats()[1].sent_wire, 0);
        } catch (...) {
            test::count_allocations = false;
            begin = true;
            release = true;
            peer.join();
            throw;
        }
        peer.join();
        if (error) {
            std::rethrow_exception(error);
        }
    }

    TEST(Net, RejectWrongLength) {
        auto c = ports();
        std::exception_ptr peer_error;
        std::thread peer([&] {
            try {
                Net net(1, c);
                std::array<std::byte, 2> data{};
                net.send(0, data);
            } catch (...) {
                peer_error = std::current_exception();
            }
        });
        {
            Net net(0, c);
            std::array<std::byte, 3> data{};
            EXPECT_THROW(net.recv(1, data), std::runtime_error);
            EXPECT_THROW(net.recv(1, data), std::runtime_error);
        }
        peer.join();
        if (peer_error) {
            std::rethrow_exception(peer_error);
        }
    }
}
