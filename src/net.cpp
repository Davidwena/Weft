#include "bj/core/multi_round.hpp"
#include "bj/net/net.hpp"
#include "bj/common/aligned.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <mutex>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace bj {
    namespace {
        using Clock = std::chrono::steady_clock;

        struct Fd {
            int value = -1;
            Fd() = default;

            explicit Fd(int v) : value(v) {
            }

            ~Fd() {
                if (value >= 0) {
                    ::close(value);
                }
            }

            Fd(Fd &&o) noexcept : value(std::exchange(o.value, -1)) {
            }

            Fd &operator=(Fd &&o) noexcept {
                if (value >= 0) {
                    ::close(value);
                }
                value = std::exchange(o.value, -1);
                return *this;
            }

            Fd(const Fd &) = delete;
        };

        [[noreturn]] void fail(const char *s) {
            throw std::runtime_error(std::string(s) + ": " + std::strerror(errno));
        }

        void wait_socket(int fd, short events, Clock::time_point deadline) {
            auto remain =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now())
                    .count();
            if (remain <= 0) {
                throw std::runtime_error("network startup timeout");
            }
            pollfd p{fd, events, 0};
            int r;
            do {
                r = ::poll(&p, 1, static_cast<int>(remain));
            } while (r < 0 && errno == EINTR);
            if (!r) {
                throw std::runtime_error("network startup timeout");
            }
            if (r < 0) {
                fail("startup poll");
            }
        }

        void socket_options(int fd) {
            int one = 1;
            if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one))) {
                fail("TCP_NODELAY");
            }

            Fd probe(::socket(AF_INET, SOCK_STREAM, 0));
            if (probe.value < 0) {
                fail("buffer probe");
            }
            constexpr int desired = 4 * 1024 * 1024;
            for (int option : {SO_SNDBUF, SO_RCVBUF}) {
                if (::setsockopt(probe.value, SOL_SOCKET, option, &desired, sizeof(desired))) {
                    fail("buffer probe option");
                }
                int actual = 0;
                socklen_t length = sizeof(actual);
                if (::getsockopt(probe.value, SOL_SOCKET, option, &actual, &length)) {
                    fail("buffer probe result");
                }
                if (actual >= 2 * desired &&
                    ::setsockopt(fd, SOL_SOCKET, option, &desired, sizeof(desired))) {
                    fail("socket buffer");
                }
            }
        }

        void nonblocking(int fd) {
            int f = ::fcntl(fd, F_GETFL);
            if (f < 0 || ::fcntl(fd, F_SETFL, f | O_NONBLOCK) < 0) {
                fail("fcntl");
            }
        }

        void handshake(int fd,
                       std::span<std::uint32_t> words,
                       bool sending,
                       Clock::time_point deadline) {
            auto *ptr = reinterpret_cast<std::byte *>(words.data());
            std::size_t done = 0, bytes = words.size_bytes();
            while (done < bytes) {
                wait_socket(fd, sending ? POLLOUT : POLLIN, deadline);
                auto r = sending ? ::send(fd, ptr + done, bytes - done, MSG_NOSIGNAL)
                                 : ::recv(fd, ptr + done, bytes - done, 0);
                if (r > 0) {
                    done += static_cast<std::size_t>(r);
                } else if (r == 0) {
                    throw std::runtime_error("peer closed during handshake");
                } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                    fail("handshake");
                }
            }
        }

        struct Address {
            addrinfo *list = nullptr;

            explicit Address(const Host &h, bool passive = false) {
                addrinfo hints{};
                hints.ai_family = AF_INET;
                hints.ai_socktype = SOCK_STREAM;
                hints.ai_flags = passive ? AI_PASSIVE : 0;
                auto port = std::to_string(h.port);
                int e = ::getaddrinfo(h.address.c_str(), port.c_str(), &hints, &list);
                if (e) {
                    throw std::runtime_error(gai_strerror(e));
                }
            }

            ~Address() {
                ::freeaddrinfo(list);
            }
        };
    }

    struct Net::Impl {
        unsigned self, n, timeout, king;
        OpenMode mode;
        OpenStats open_counters;
        IoTiming timing;
        OpenBuf *reducing = nullptr;
        std::vector<AlignedBuffer<std::byte>> scratch;
        std::vector<Fd> sockets;
        std::vector<PeerStats> counters;
        std::uint64_t round_count = 0;

        struct Transfer {
            std::span<const std::byte> tx;
            std::span<std::byte> rx;
            std::array<std::uint64_t, 2> tx_header{}, rx_header{};
            std::size_t header_bytes = 8, rx_bytes = 0, chunk = 0, chunk_received = 0;
            std::uint64_t fingerprint = 0;
            bool streamed = false;

            std::size_t receive_bytes() const noexcept {
                return streamed ? rx_bytes : rx.size();
            }

            std::size_t sent = 0, received = 0;
            bool sending = false, receiving = false;
        };

        std::vector<Transfer> transfers;
        std::vector<pollfd> polls;
        std::mutex mutex;
        std::condition_variable ready, finished;
        bool pending = false, done = false, stop = false, broken = false;
        std::exception_ptr error;
        std::thread worker;

        Impl(unsigned id, const Config &config)
            : self(id), n(static_cast<unsigned>(config.hosts.size())), timeout(config.timeout_ms),
              king(config.king), mode(config.open_mode), sockets(n), counters(n), transfers(n),
              polls(n) {
            config.validate(n);
            if (self >= n) {
                throw std::invalid_argument("party id out of range");
            }
            scratch.reserve(n);
            for (unsigned p = 0; p < n; ++p) {
                scratch.emplace_back(p == self ? 0 : OpenLayout::chunk_bytes);
            }
            const auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
            Address address(config.hosts[self], true);
            Fd listener(::socket(AF_INET, SOCK_STREAM, 0));
            if (listener.value < 0) {
                fail("socket");
            }
            int one = 1;
            if (::setsockopt(listener.value, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one))) {
                fail("SO_REUSEADDR");
            }
            if (::bind(listener.value, address.list->ai_addr, address.list->ai_addrlen)) {
                fail("bind");
            }
            if (::listen(listener.value, static_cast<int>(n))) {
                fail("listen");
            }
            nonblocking(listener.value);
            for (unsigned p = 0; p < self; ++p) {
                Address peer(config.hosts[p]);
                for (;;) {
                    Fd fd(::socket(AF_INET, SOCK_STREAM, 0));
                    if (fd.value < 0) {
                        fail("socket");
                    }
                    nonblocking(fd.value);
                    int r = ::connect(fd.value, peer.list->ai_addr, peer.list->ai_addrlen);
                    int e = errno;
                    if (r < 0 && e == EINPROGRESS) {
                        wait_socket(fd.value, POLLOUT, deadline);
                        socklen_t len = sizeof(e);
                        if (::getsockopt(fd.value, SOL_SOCKET, SO_ERROR, &e, &len)) {
                            fail("connect status");
                        }
                    } else if (!r) {
                        e = 0;
                    }
                    if (!e) {
                        socket_options(fd.value);
                        sockets[p] = std::move(fd);
                        break;
                    }
                    if (e != ECONNREFUSED && e != ETIMEDOUT && e != EHOSTUNREACH &&
                        e != ENETUNREACH) {
                        errno = e;
                        fail("connect");
                    }
                    if (Clock::now() >= deadline) {
                        throw std::runtime_error("connect timeout");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                std::array<std::uint32_t, 5> hello{htonl(0x424a0002),
                                                   htonl(n),
                                                   htonl(self),
                                                   htonl(king),
                                                   htonl(static_cast<unsigned>(mode))};
                handshake(sockets[p].value, hello, true, deadline);
            }
            for (unsigned count = self + 1; count < n; ++count) {
                wait_socket(listener.value, POLLIN, deadline);
                Fd fd(::accept(listener.value, nullptr, nullptr));
                if (fd.value < 0) {
                    fail("accept");
                }
                nonblocking(fd.value);
                socket_options(fd.value);
                std::array<std::uint32_t, 5> hello{};
                handshake(fd.value, hello, false, deadline);
                const auto p = ntohl(hello[2]);
                if (ntohl(hello[0]) != 0x424a0002 || ntohl(hello[1]) != n ||
                    ntohl(hello[3]) != king || ntohl(hello[4]) != static_cast<unsigned>(mode) ||
                    p <= self || p >= n || sockets[p].value >= 0) {
                    throw std::runtime_error("invalid peer handshake");
                }
                sockets[p] = std::move(fd);
            }
            worker = std::thread([this] {
                run();
            });
        }

        ~Impl() {
            {
                std::lock_guard lock(mutex);
                stop = true;
            }
            ready.notify_one();
            if (worker.joinable()) {
                worker.join();
            }
        }

        void clear_job() noexcept {
            for (auto &t : transfers) {
                t = Transfer{};
            }
        }

        void submit() {
            const auto blocked_start = Clock::now();
            std::unique_lock lock(mutex);
            if (broken) {
                throw std::runtime_error("network unusable after failed transfer");
            }
            done = false;
            pending = true;
            error = nullptr;
            ready.notify_one();
            finished.wait(lock, [this] {
                return done;
            });
            timing.caller_blocked_seconds +=
                std::chrono::duration<double>(Clock::now() - blocked_start).count();
            if (error) {
                broken = true;
                std::rethrow_exception(error);
            }
            for (unsigned p = 0; p < n; ++p) {
                const auto &t = transfers[p];
                auto &c = counters[p];
                if (t.sending) {
                    c.sent_payload += t.tx.size();
                    c.sent_wire += t.tx.size() + t.header_bytes;
                    ++c.sent_messages;
                }
                if (t.receiving) {
                    c.received_payload += t.receive_bytes();
                    c.received_wire += t.receive_bytes() + t.header_bytes;
                    ++c.received_messages;
                }
            }
        }

        void run() noexcept {
            for (;;) {
                {
                    std::unique_lock lock(mutex);
                    ready.wait(lock, [this] {
                        return pending || stop;
                    });
                    if (stop) {
                        return;
                    }
                    pending = false;
                }
                std::exception_ptr caught;
                try {
                    transfer();
                } catch (...) {
                    caught = std::current_exception();
                }
                {
                    std::lock_guard lock(mutex);
                    error = caught;
                    done = true;
                }
                finished.notify_one();
            }
        }

        void transfer() {
            const auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
            for (auto &t : transfers) {
                t.tx_header[0] = __builtin_bswap64(t.tx.size());
                t.tx_header[1] = __builtin_bswap64(t.fingerprint);
            }
            for (;;) {
                unsigned active = 0;
                for (unsigned p = 0; p < n; ++p) {
                    const auto &t = transfers[p];
                    short events = 0;
                    if (t.sending && t.sent < t.tx.size() + t.header_bytes) {
                        events |= POLLOUT;
                    }
                    if (t.receiving && t.received < t.receive_bytes() + t.header_bytes) {
                        events |= POLLIN;
                    }
                    polls[p] = {events ? sockets[p].value : -1, events, 0};
                    active += (events != 0);
                }
                if (!active) {
                    return;
                }
                const auto remain =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now())
                        .count();
                if (remain <= 0) {
                    throw std::runtime_error("network transfer timeout");
                }
                const auto poll_start = Clock::now();
                int r = ::poll(polls.data(), n, static_cast<int>(remain));
                timing.poll_wait_seconds +=
                    std::chrono::duration<double>(Clock::now() - poll_start).count();
                if (r < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    fail("poll");
                }
                if (!r) {
                    throw std::runtime_error("network transfer timeout");
                }
                for (unsigned p = 0; p < n; ++p) {
                    const short ev = polls[p].revents;
                    if (!ev) {
                        continue;
                    }
                    auto &t = transfers[p];
                    int fd = sockets[p].value;
                    if (ev & POLLNVAL) {
                        throw std::runtime_error("invalid socket");
                    }
                    if ((ev & POLLOUT) && t.sending && t.sent < t.tx.size() + t.header_bytes) {
                        iovec iov[2]{};
                        unsigned count = 0;
                        if (t.sent < t.header_bytes) {
                            iov[count++] = {reinterpret_cast<std::byte *>(t.tx_header.data()) +
                                                t.sent,
                                            t.header_bytes - t.sent};
                            if (!t.tx.empty()) {
                                iov[count++] = {const_cast<std::byte *>(t.tx.data()), t.tx.size()};
                            }
                        } else {
                            iov[count++] = {const_cast<std::byte *>(t.tx.data()) +
                                                (t.sent - t.header_bytes),
                                            t.tx.size() - (t.sent - t.header_bytes)};
                        }
                        msghdr msg{};
                        msg.msg_iov = iov;
                        msg.msg_iovlen = count;
                        const auto io_start = Clock::now();
                        auto bytes = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
                        timing.socket_io_seconds +=
                            std::chrono::duration<double>(Clock::now() - io_start).count();
                        if (bytes > 0) {
                            t.sent += static_cast<std::size_t>(bytes);
                        } else if (bytes < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                                   errno != EINTR) {
                            fail("sendmsg");
                        }
                    }
                    if ((ev & (POLLIN | POLLHUP | POLLERR)) && t.receiving &&
                        t.received < t.receive_bytes() + t.header_bytes) {
                        void *data;
                        std::size_t length;
                        if (t.received < t.header_bytes) {
                            data = reinterpret_cast<std::byte *>(t.rx_header.data()) + t.received;
                            length = t.header_bytes - t.received;
                        } else if (t.streamed) {
                            const auto &chunk = reducing->layout.chunks()[t.chunk];
                            data = scratch[p].data() + t.chunk_received;
                            length = chunk.bytes - t.chunk_received;
                        } else {
                            data = t.rx.data() + (t.received - t.header_bytes);
                            length = t.rx.size() - (t.received - t.header_bytes);
                        }
                        const bool body = t.received >= t.header_bytes;
                        const auto io_start = Clock::now();
                        auto bytes = ::recv(fd, data, length, 0);
                        timing.socket_io_seconds +=
                            std::chrono::duration<double>(Clock::now() - io_start).count();
                        if (bytes > 0) {
                            t.received += static_cast<std::size_t>(bytes);
                            if (!body && t.received == t.header_bytes) {
                                if (__builtin_bswap64(t.rx_header[0]) != t.receive_bytes()) {
                                    throw std::runtime_error(
                                        "frame length does not match receive buffer");
                                }
                                if (t.header_bytes == 16 &&
                                    __builtin_bswap64(t.rx_header[1]) != t.fingerprint) {
                                    throw std::runtime_error("open layout mismatch");
                                }
                            }
                            if (body && t.streamed) {
                                t.chunk_received += static_cast<std::size_t>(bytes);
                                const auto &chunk = reducing->layout.chunks()[t.chunk];
                                if (t.chunk_received == chunk.bytes) {
                                    reducing->accumulate(t.chunk,
                                                         scratch[p].span().first(chunk.bytes));
                                    ++open_counters.reduced_chunks;
                                    const bool incomplete = std::any_of(
                                        transfers.begin(),
                                        transfers.end(),
                                        [](const Transfer &other) {
                                            return other.receiving &&
                                                   other.received <
                                                       other.receive_bytes() + other.header_bytes;
                                        });
                                    open_counters.chunks_before_all_received += incomplete;
                                    ++t.chunk;
                                    t.chunk_received = 0;
                                }
                            }
                        } else if (!bytes) {
                            throw std::runtime_error("peer closed before frame completed");
                        } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                            fail("recv");
                        }
                    } else if (ev & (POLLERR | POLLHUP)) {
                        throw std::runtime_error("peer connection failed");
                    }
                }
            }
        }
    };

    Net::Net(unsigned id, const Config &config) : impl_(std::make_unique<Impl>(id, config)) {
    }

    Net::~Net() = default;

    unsigned Net::id() const noexcept {
        return impl_->self;
    }

    unsigned Net::size() const noexcept {
        return impl_->n;
    }

    const std::vector<PeerStats> &Net::stats() const noexcept {
        return impl_->counters;
    }

    std::uint64_t Net::rounds() const noexcept {
        return impl_->round_count;
    }

    IoTiming Net::io_timing() const noexcept {
        return impl_->timing;
    }

    void Net::reset_stats() noexcept {
        impl_->timing = IoTiming{};
        std::fill(impl_->counters.begin(), impl_->counters.end(), PeerStats{});
        impl_->round_count = 0;
        impl_->open_counters = OpenStats{};
    }

    unsigned Net::king() const noexcept {
        return impl_->king;
    }

    OpenMode Net::open_mode() const noexcept {
        return impl_->mode;
    }

    bool Net::uses_king() const noexcept {
        return size() >= 3 && open_mode() == OpenMode::Auto;
    }

    namespace {
        thread_local NetworkHook *net_hook = nullptr;
    }

    NetworkHook *Net::thread_hook() noexcept {
        return net_hook;
    }

    void Net::thread_hook(NetworkHook *hook) noexcept {
        net_hook = hook;
    }

    const OpenStats &Net::open_stats() const noexcept {
        return net_hook ? net_hook->counters : impl_->open_counters;
    }

    std::size_t Net::receive_scratch_bytes() const noexcept {
        return (size() - 1) * OpenLayout::chunk_bytes;
    }

    void Net::open_round(OpenBuf &buf) {
        if (net_hook) {
            net_hook->open(buf);
            return;
        }
        buf.initialize();
        impl_->reducing = &buf;
        const auto bytes = buf.layout.bytes(), tag = buf.layout.fingerprint();
        auto configure =
            [&](unsigned p, bool send, bool receive, bool streamed, std::span<const std::byte> tx) {
                auto &t = impl_->transfers[p];
                t.sending = send;
                t.receiving = receive;
                t.tx = tx;
                t.header_bytes = 16;
                t.fingerprint = tag;
                if (streamed) {
                    t.streamed = true;
                    t.rx_bytes = bytes;
                } else {
                    t.rx = buf.opened;
                }
            };
        impl_->clear_job();
        if (!uses_king()) {
            for (unsigned p = 0; p < size(); ++p) {
                if (p != id()) {
                    configure(p, true, true, true, buf.share);
                }
            }
            impl_->submit();
        } else if (id() != king()) {
            configure(king(), true, true, false, buf.share);
            impl_->submit();
        } else {
            for (unsigned p = 0; p < size(); ++p) {
                if (p != id()) {
                    configure(p, false, true, true, {});
                }
            }
            impl_->submit();
            impl_->clear_job();
            for (unsigned p = 0; p < size(); ++p) {
                if (p != id()) {
                    configure(p, true, false, false, buf.opened);
                }
            }
            impl_->submit();
        }
        impl_->reducing = nullptr;
        ++impl_->open_counters.rounds;
        impl_->open_counters.hops += uses_king() ? 2 : 1;
        impl_->open_counters.contribution_bits += buf.layout.bits();
    }

    void Net::reveal_to(unsigned recipient, OpenBuf &buf) {
        if (recipient >= size()) {
            throw std::invalid_argument("private output recipient");
        }
        impl_->clear_job();
        impl_->reducing = &buf;
        if (id() == recipient) {
            buf.initialize();
            for (unsigned p = 0; p < size(); ++p) {
                if (p != id()) {
                    auto &t = impl_->transfers[p];
                    t.receiving = true;
                    t.streamed = true;
                    t.rx_bytes = buf.layout.bytes();
                    t.header_bytes = 16;
                    t.fingerprint = buf.layout.fingerprint();
                }
            }
        } else {
            auto &t = impl_->transfers[recipient];
            t.sending = true;
            t.tx = buf.share;
            t.header_bytes = 16;
            t.fingerprint = buf.layout.fingerprint();
        }

        const auto saved = impl_->open_counters;
        impl_->submit();
        impl_->open_counters = saved;
        impl_->reducing = nullptr;
    }

    void Net::exchange_all(std::span<const std::span<const std::byte>> tx,
                           std::span<const std::span<std::byte>> rx) {
        if (tx.size() != size() || rx.size() != size()) {
            throw std::invalid_argument("exchange span count must be n");
        }
        if (!tx[id()].empty() || !rx[id()].empty()) {
            throw std::invalid_argument("self spans must be empty");
        }
        impl_->clear_job();
        for (unsigned p = 0; p < size(); ++p) {
            if (p != id()) {
                auto &t = impl_->transfers[p];
                t.tx = tx[p];
                t.rx = rx[p];
                t.sending = true;
                t.receiving = true;
            }
        }
        impl_->submit();
        ++impl_->round_count;
    }

    void Net::send(unsigned peer, std::span<const std::byte> data) {
        if (peer >= size() || peer == id()) {
            throw std::invalid_argument("invalid send peer");
        }
        impl_->clear_job();
        auto &t = impl_->transfers[peer];
        t.tx = data;
        t.sending = true;
        impl_->submit();
    }

    void Net::recv(unsigned peer, std::span<std::byte> data) {
        if (peer >= size() || peer == id()) {
            throw std::invalid_argument("invalid recv peer");
        }
        impl_->clear_job();
        auto &t = impl_->transfers[peer];
        t.rx = data;
        t.receiving = true;
        impl_->submit();
    }

    void Net::broadcast_from(unsigned sender, std::span<std::byte> data) {
        if (sender >= size()) {
            throw std::invalid_argument("invalid broadcast sender");
        }
        impl_->clear_job();
        if (id() == sender) {
            for (unsigned peer = 0; peer < size(); ++peer) {
                if (peer != sender) {
                    auto &t = impl_->transfers[peer];
                    t.tx = data;
                    t.sending = true;
                }
            }
        } else {
            auto &t = impl_->transfers[sender];
            t.rx = data;
            t.receiving = true;
        }
        impl_->submit();
    }

}
