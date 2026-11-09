#include "bj/net/open_buf.hpp"
#include "bj/common/packed.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace bj {
    void OpenLayout::add(unsigned width, std::size_t count) {
        if (sealed_) {
            throw std::logic_error("open layout already sealed");
        }
        const auto k = width ? width : 1;
        if (count > (std::numeric_limits<std::size_t>::max() - 7) / k) {
            throw std::length_error("open field too large");
        }
        const auto bits = count * k, bytes = (bits + 7) / 8;
        if (bytes > std::numeric_limits<std::size_t>::max() - bytes_ || bits > UINT64_MAX - bits_) {
            throw std::length_error("open layout too large");
        }
        fields_.push_back({width, count, bytes_, bytes});
        bytes_ += bytes;
        bits_ += bits;
        for (auto value : {std::uint64_t(width), std::uint64_t(count)}) {
            for (unsigned i = 0; i < 8; ++i) {
                fingerprint_ ^= (value >> (i * 8)) & 255;
                fingerprint_ *= 1099511628211ULL;
            }
        }
    }

    OpenLayout &OpenLayout::add_ring(unsigned width, std::size_t count) {
        if (width < 1 || width > 96) {
            throw std::invalid_argument("ring width must be 1..96");
        }
        add(width, count);
        return *this;
    }

    OpenLayout &OpenLayout::add_bits(std::size_t count) {
        add(0, count);
        return *this;
    }

    void OpenLayout::seal(bool resizable) {
        if (sealed_) {
            return;
        }
        for (std::size_t i = 0; i < fields_.size(); ++i) {
            const auto &f = fields_[i];
            const auto k = f.width ? f.width : 1;

            const auto step = (chunk_bytes / k) * k, per_chunk = (step * 8) / k;
            for (std::size_t offset = 0, elements = 0; offset < f.bytes;
                 offset += step, elements += per_chunk) {
                chunks_.push_back({i,
                                   f.offset + offset,
                                   std::min(step, f.bytes - offset),
                                   std::min(per_chunk, f.count - elements)});
            }
        }
        sealed_ = true;
        resizable_ = resizable;
        if (resizable) {
            for (const auto &f : fields_) {
                limits_.push_back(f.count);
            }
        }
    }

    void OpenLayout::resize_counts(std::span<const std::size_t> counts) {
        if (!sealed_ || !resizable_ || counts.size() != fields_.size()) {
            throw std::logic_error("open layout is not bounded-resizable");
        }
        for (std::size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > limits_[i]) {
                throw std::length_error("open batch exceeds offline bound");
            }
        }
        bytes_ = 0;
        bits_ = 0;
        fingerprint_ = 14695981039346656037ULL;
        chunks_.clear();
        for (std::size_t i = 0; i < fields_.size(); ++i) {
            auto &f = fields_[i];
            f.count = counts[i];
            f.offset = bytes_;
            const auto k = f.width ? f.width : 1;
            f.bytes = (f.count * k + 7) / 8;
            bytes_ += f.bytes;
            bits_ += f.count * k;
            for (auto v : {std::uint64_t(f.width), std::uint64_t(f.count)}) {
                for (unsigned b = 0; b < 8; ++b) {
                    fingerprint_ ^= (v >> (b * 8)) & 255;
                    fingerprint_ *= 1099511628211ULL;
                }
            }
            const auto step = (chunk_bytes / k) * k, per_chunk = step * 8 / k;
            for (std::size_t off = 0, e = 0; off < f.bytes; off += step, e += per_chunk) {
                chunks_.push_back({i,
                                   f.offset + off,
                                   std::min(step, f.bytes - off),
                                   std::min(per_chunk, f.count - e)});
            }
        }
    }

    void OpenLayout::reserve_batch(std::size_t fields, std::size_t bytes) {
        if (sealed_ || !fields || bytes > SIZE_MAX / 2) {
            throw std::invalid_argument("batch bounds");
        }
        fields_.reserve(fields);
        chunks_.reserve(fields + 2 * bytes / chunk_bytes + 1);
    }

    void OpenLayout::configure_batch(std::span<const OpenField> fields) {
        if (fields.size() > fields_.capacity()) {
            throw std::length_error("merged open field bound");
        }
        fields_.clear();
        chunks_.clear();
        sealed_ = false;
        bytes_ = 0;
        bits_ = 0;
        fingerprint_ = 14695981039346656037ULL;
        for (const auto &f : fields) {
            if (f.width > 96) {
                throw std::invalid_argument("merged open width");
            }
            add(f.width, f.count);
        }
        const auto need = fields_.size() + 2 * bytes_ / chunk_bytes + 1;
        if (need > chunks_.capacity()) {
            throw std::length_error("merged open chunk bound");
        }
        seal();
    }

    OpenBuf::OpenBuf(const OpenLayout &l, std::span<const std::byte> s, std::span<std::byte> o)
        : layout(l), share(s), opened(o) {
        if (!l.sealed() || s.size() != l.bytes() || o.size() != l.bytes()) {
            throw std::invalid_argument("open buffer/schema mismatch");
        }
        const auto a = reinterpret_cast<std::uintptr_t>(s.data()),
                   b = reinterpret_cast<std::uintptr_t>(o.data());
        if (!s.empty() && (a <= b ? b - a < s.size() : a - b < o.size())) {
            throw std::invalid_argument("open input/output must not overlap");
        }
    }

    void OpenBuf::initialize() noexcept {
        if (!share.empty()) {
            std::memcpy(opened.data(), share.data(), share.size());
        }
        for (const auto &f : layout.fields()) {
            const auto used = (f.count % 8) * (f.width ? f.width : 1) % 8;
            if (used) {
                opened[f.offset + f.bytes - 1] &= std::byte((1u << used) - 1);
            }
        }
    }

    namespace {
        template <class T>
        void sum_native(std::span<std::byte> out,
                        std::span<const std::byte> in,
                        std::size_t count) noexcept {
            for (std::size_t i = 0; i < count; ++i) {
                T a, b;
                std::memcpy(&a, out.data() + i * sizeof(T), sizeof(T));
                std::memcpy(&b, in.data() + i * sizeof(T), sizeof(T));
                a += b;
                std::memcpy(out.data() + i * sizeof(T), &a, sizeof(T));
            }
        }

        std::uint64_t read_bits(const std::byte *data, std::size_t bit, unsigned k) noexcept {
            std::uint64_t x = 0;
            for (unsigned used = 0; used < k;) {
                const unsigned shift = static_cast<unsigned>(bit % 8),
                               take = std::min(8 - shift, k - used);
                x |= std::uint64_t((std::to_integer<unsigned>(data[bit / 8]) >> shift) &
                                   ((1u << take) - 1))
                     << used;
                used += take;
                bit += take;
            }
            return x;
        }

        void write_bits(std::byte *data, std::size_t bit, unsigned k, std::uint64_t x) noexcept {
            for (unsigned used = 0; used < k;) {
                const unsigned shift = static_cast<unsigned>(bit % 8),
                               take = std::min(8 - shift, k - used),
                               mask = ((1u << take) - 1) << shift;
                const auto prev = std::to_integer<unsigned>(data[bit / 8]);
                data[bit / 8] = std::byte((prev & ~mask) |
                                          ((static_cast<unsigned>(x >> used) << shift) & mask));
                used += take;
                bit += take;
            }
        }
    }

    void OpenBuf::accumulate(std::size_t index, std::span<const std::byte> data) noexcept {
        const auto &chunk = layout.chunks()[index];
        const auto &field = layout.fields()[chunk.field];
        auto out = opened.subspan(chunk.offset, chunk.bytes);
        if (!field.width) {
            std::size_t i = 0;
            for (; i + 8 <= data.size(); i += 8) {
                std::uint64_t a, b;
                std::memcpy(&a, out.data() + i, 8);
                std::memcpy(&b, data.data() + i, 8);
                a ^= b;
                std::memcpy(out.data() + i, &a, 8);
            }
            for (; i < data.size(); ++i) {
                out[i] ^= data[i];
            }
        } else if (field.width == 32) {
            sum_native<std::uint32_t>(out, data, chunk.count);
        } else if (field.width == 64) {
            sum_native<std::uint64_t>(out, data, chunk.count);
        } else if (field.width > 64) {
            for (std::size_t i = 0; i < chunk.count; ++i) {
                store_wide(out,
                           i * field.width,
                           field.width,
                           (load_wide(out, i * field.width, field.width) +
                            load_wide(data, i * field.width, field.width)) &
                               native_mask<uint128_t>(field.width));
            }
        } else {
            const auto k = field.width;
            const auto mask = UINT64_MAX >> (64 - k);
            for (std::size_t i = 0; i < chunk.count; ++i) {
                const auto bit = i * k;
                const auto x =
                    (read_bits(out.data(), bit, k) + read_bits(data.data(), bit, k)) & mask;
                write_bits(out.data(), bit, k, x);
            }
        }
        const auto used = (chunk.count % 8) * (field.width ? field.width : 1) % 8;
        if (used) {
            out.back() &= std::byte((1u << used) - 1);
        }
    }
}
