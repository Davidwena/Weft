#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bj {

    struct OpenField {
        unsigned width;
        std::size_t count, offset, bytes;
    };

    struct OpenChunk {
        std::size_t field, offset, bytes, count;
    };

    class OpenLayout {
        std::vector<OpenField> fields_;
        std::vector<OpenChunk> chunks_;
        std::size_t bytes_ = 0;
        std::uint64_t bits_ = 0, fingerprint_ = 14695981039346656037ULL;
        bool sealed_ = false, resizable_ = false;
        std::vector<std::size_t> limits_;
        void add(unsigned width, std::size_t count);

    public:
        static constexpr std::size_t chunk_bytes = 64 * 1024;
        OpenLayout &add_ring(unsigned width, std::size_t count);
        OpenLayout &add_bits(std::size_t count);
        void seal(bool resizable = false);
        void reserve_batch(std::size_t max_fields, std::size_t max_bytes);
        void configure_batch(std::span<const OpenField> fields);
        void resize_counts(std::span<const std::size_t>);

        bool sealed() const noexcept {
            return sealed_;
        }

        std::size_t bytes() const noexcept {
            return bytes_;
        }

        std::uint64_t bits() const noexcept {
            return bits_;
        }

        std::uint64_t fingerprint() const noexcept {
            return fingerprint_;
        }

        std::span<const OpenField> fields() const noexcept {
            return fields_;
        }

        std::span<const OpenChunk> chunks() const noexcept {
            return chunks_;
        }
    };

    struct OpenBuf {
        const OpenLayout &layout;
        std::span<const std::byte> share;
        std::span<std::byte> opened;
        OpenBuf(const OpenLayout &layout,
                std::span<const std::byte> share,
                std::span<std::byte> opened);
        void initialize() noexcept;

        void accumulate(std::size_t chunk, std::span<const std::byte> data) noexcept;
    };
}
