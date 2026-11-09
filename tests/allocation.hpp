#pragma once
#include <atomic>
#include <cstdint>

namespace bj::test {
    extern std::atomic<bool> count_allocations;
    extern std::atomic<std::uint64_t> allocations;
}
