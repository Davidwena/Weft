#pragma once
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <utility>

namespace bj {

    template <class T> class AlignedBuffer {
        static_assert(std::is_trivially_copyable_v<T>);

        struct Free {
            void operator()(T *p) const noexcept {
                std::free(p);
            }
        };

        std::unique_ptr<T, Free> data_;
        std::size_t size_ = 0;

    public:
        AlignedBuffer() = default;

        explicit AlignedBuffer(std::size_t n) : size_(n) {
            if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw std::length_error("buffer too large");
            }
            if (n) {
                void *p = nullptr;
                if (posix_memalign(&p, 64, n * sizeof(T))) {
                    throw std::bad_alloc();
                }
                data_.reset(static_cast<T *>(p));
            }
        }

        AlignedBuffer(AlignedBuffer &&other) noexcept
            : data_(std::move(other.data_)), size_(std::exchange(other.size_, 0)) {
        }

        AlignedBuffer &operator=(AlignedBuffer &&other) noexcept {
            if (this != &other) {
                data_ = std::move(other.data_);
                size_ = std::exchange(other.size_, 0);
            }
            return *this;
        }

        AlignedBuffer(const AlignedBuffer &) = delete;
        AlignedBuffer &operator=(const AlignedBuffer &) = delete;

        T *data() noexcept {
            return data_.get();
        }

        const T *data() const noexcept {
            return data_.get();
        }

        std::size_t size() const noexcept {
            return size_;
        }

        std::span<T> span() noexcept {
            return {data(), size_};
        }

        std::span<const T> span() const noexcept {
            return {data(), size_};
        }
    };

    class BufferPool {
        std::vector<AlignedBuffer<std::byte>> slots_;

    public:
        BufferPool(std::size_t slots, std::size_t bytes) {
            slots_.reserve(slots);
            for (std::size_t i = 0; i < slots; ++i) {
                slots_.emplace_back(bytes);
            }
        }

        std::span<std::byte> get(std::size_t slot, std::size_t bytes) {
            auto &b = slots_.at(slot);
            if (bytes > b.size()) {
                throw std::length_error("pool capacity exceeded");
            }
            return b.span().first(bytes);
        }
    };
}
