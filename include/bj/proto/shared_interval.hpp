#pragma once
#include "bj/plan/investigation.hpp"
#include <functional>

namespace bj {
    struct SharedIntervalHandle {
        bool requested = false, done = false;
    };

    using IntervalBinder =
        std::function<SharedIntervalHandle *(std::size_t,
                                             unsigned,
                                             const void *,
                                             std::span<const BandWindow>,
                                             std::span<const std::span<std::uint64_t>>)>;
    IntervalBinder *interval_binder() noexcept;
    void interval_binder(IntervalBinder *) noexcept;
}
