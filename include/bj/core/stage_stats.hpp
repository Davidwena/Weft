#pragma once
#include "bj/net/net.hpp"

namespace bj {
    struct StageStats {
        double seconds = 0;
        OpenStats open{};
        IoTiming io{};
        std::vector<PeerStats> peers;

        explicit StageStats(unsigned n) : peers(n) {
        }

        void capture(const Net &net, double time, const StageStats *before = nullptr) {
            seconds = time;
            open = net.open_stats();
            io = net.io_timing();
            if (before) {
                open.rounds -= before->open.rounds;
                open.hops -= before->open.hops;
                open.contribution_bits -= before->open.contribution_bits;
                io.poll_wait_seconds -= before->io.poll_wait_seconds;
                io.socket_io_seconds -= before->io.socket_io_seconds;
                io.caller_blocked_seconds -= before->io.caller_blocked_seconds;
            }
            for (std::size_t i = 0; i < peers.size(); ++i) {
                const auto &p = net.stats()[i];
                const PeerStats z{};
                const auto &b = before ? before->peers[i] : z;
                peers[i] = {p.sent_payload - b.sent_payload,
                            p.received_payload - b.received_payload,
                            p.sent_wire - b.sent_wire,
                            p.received_wire - b.received_wire,
                            p.sent_messages - b.sent_messages,
                            p.received_messages - b.received_messages};
            }
        }
    };
}
