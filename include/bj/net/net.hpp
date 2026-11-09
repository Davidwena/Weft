#pragma once
#include "bj/common/config.hpp"
#include "bj/net/open_buf.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace bj {
    struct NetworkHook;

    struct PeerStats {
        std::uint64_t sent_payload = 0, received_payload = 0;
        std::uint64_t sent_wire = 0, received_wire = 0;
        std::uint64_t sent_messages = 0, received_messages = 0;
    };

    struct OpenStats {
        std::uint64_t rounds = 0, hops = 0, contribution_bits = 0;
        std::uint64_t reduced_chunks = 0, chunks_before_all_received = 0;
    };

    struct IoTiming {
        double caller_blocked_seconds = 0, poll_wait_seconds = 0, socket_io_seconds = 0;
    };

    class Net {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        static NetworkHook *thread_hook() noexcept;
        static void thread_hook(NetworkHook *) noexcept;
        Net(unsigned id, const Config &config);
        ~Net();
        Net(const Net &) = delete;
        Net &operator=(const Net &) = delete;
        unsigned id() const noexcept;
        unsigned size() const noexcept;

        void open_round(OpenBuf &buf);
        void reveal_to(unsigned recipient, OpenBuf &buf);
        unsigned king() const noexcept;
        OpenMode open_mode() const noexcept;
        bool uses_king() const noexcept;
        const OpenStats &open_stats() const noexcept;
        std::size_t receive_scratch_bytes() const noexcept;
        void exchange_all(std::span<const std::span<const std::byte>> send_bufs,
                          std::span<const std::span<std::byte>> recv_bufs);
        void send(unsigned peer, std::span<const std::byte> data);
        void recv(unsigned peer, std::span<std::byte> data);
        void broadcast_from(unsigned sender, std::span<std::byte> data);
        const std::vector<PeerStats> &stats() const noexcept;
        std::uint64_t rounds() const noexcept;
        IoTiming io_timing() const noexcept;
        void reset_stats() noexcept;
    };
}
