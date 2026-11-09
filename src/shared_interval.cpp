#include "bj/proto/shared_interval.hpp"

namespace bj {
    namespace {
        thread_local IntervalBinder *binder = nullptr;
    }

    IntervalBinder *interval_binder() noexcept {
        return binder;
    }

    void interval_binder(IntervalBinder *b) noexcept {
        binder = b;
    }
}
