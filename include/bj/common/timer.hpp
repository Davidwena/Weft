#pragma once
#include <array>
#include <chrono>
#include <cstdint>

namespace bj {
    class Timer {
        using Clock = std::chrono::steady_clock;
        Clock::time_point start_ = Clock::now();

    public:
        void reset() noexcept {
            start_ = Clock::now();
        }

        double seconds() const noexcept {
            return std::chrono::duration<double>(Clock::now() - start_).count();
        }
    };
    enum class Phase : unsigned {
        Plan,
        Dealer,
        MaterialLoad,
        OfflineLocal,
        Input,
        Online,
        Count
    };

    struct PhaseStat {
        double seconds = 0;
        std::uint64_t sent = 0, received = 0;
    };

    class PhaseStats {
        std::array<PhaseStat, static_cast<unsigned>(Phase::Count)> stats_{};

    public:
        PhaseStat &operator[](Phase p) noexcept {
            return stats_[static_cast<unsigned>(p)];
        }

        const PhaseStat &operator[](Phase p) const noexcept {
            return stats_[static_cast<unsigned>(p)];
        }

        void
        add(Phase p, double seconds, std::uint64_t sent = 0, std::uint64_t received = 0) noexcept {
            auto &s = (*this)[p];
            s.seconds += seconds;
            s.sent += sent;
            s.received += received;
        }
    };
}
