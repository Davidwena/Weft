#include "allocation.hpp"
#include <cstdlib>
#include <new>

namespace bj::test {
    std::atomic<bool> count_allocations{false};
    std::atomic<std::uint64_t> allocations{0};
}

namespace {
    void counted() {
        if (bj::test::count_allocations.load(std::memory_order_relaxed)) {
            bj::test::allocations.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void *alloc(std::size_t n) {
        counted();
        if (auto *p = std::malloc(n ? n : 1)) {
            return p;
        }
        throw std::bad_alloc();
    }

    void *aligned(std::size_t n, std::size_t alignment) {
        counted();
        void *p = nullptr;
        if (posix_memalign(&p, alignment, n ? n : 1)) {
            throw std::bad_alloc();
        }
        return p;
    }
}

void *operator new(std::size_t n) {
    return alloc(n);
}

void *operator new[](std::size_t n) {
    return alloc(n);
}

void operator delete(void *p) noexcept {
    std::free(p);
}

void operator delete[](void *p) noexcept {
    std::free(p);
}

void operator delete(void *p, std::size_t) noexcept {
    std::free(p);
}

void operator delete[](void *p, std::size_t) noexcept {
    std::free(p);
}

void *operator new(std::size_t n, std::align_val_t a) {
    return aligned(n, static_cast<std::size_t>(a));
}

void *operator new[](std::size_t n, std::align_val_t a) {
    return aligned(n, static_cast<std::size_t>(a));
}

void operator delete(void *p, std::align_val_t) noexcept {
    std::free(p);
}

void operator delete[](void *p, std::align_val_t) noexcept {
    std::free(p);
}

void operator delete(void *p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}

void operator delete[](void *p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}

void *operator new(std::size_t n, const std::nothrow_t &) noexcept {
    try {
        return alloc(n);
    } catch (...) {
        return nullptr;
    }
}

void *operator new[](std::size_t n, const std::nothrow_t &) noexcept {
    try {
        return alloc(n);
    } catch (...) {
        return nullptr;
    }
}

void operator delete(void *p, const std::nothrow_t &) noexcept {
    std::free(p);
}

void operator delete[](void *p, const std::nothrow_t &) noexcept {
    std::free(p);
}

void *operator new(std::size_t n, std::align_val_t a, const std::nothrow_t &) noexcept {
    try {
        return aligned(n, static_cast<std::size_t>(a));
    } catch (...) {
        return nullptr;
    }
}

void *operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t &) noexcept {
    try {
        return aligned(n, static_cast<std::size_t>(a));
    } catch (...) {
        return nullptr;
    }
}

void operator delete(void *p, std::align_val_t, const std::nothrow_t &) noexcept {
    std::free(p);
}

void operator delete[](void *p, std::align_val_t, const std::nothrow_t &) noexcept {
    std::free(p);
}
