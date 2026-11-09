#include "bj/core/input.hpp"
#include "bj/data/bank.hpp"
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bj {
    namespace {
        std::uint64_t u64(const std::byte *p) {
            std::uint64_t v;
            std::memcpy(&v, p, 8);
            return v;
        }

        std::uint32_t u32(const std::byte *p) {
            std::uint32_t v;
            std::memcpy(&v, p, 4);
            return v;
        }
    }

    BankData::BankData(const std::filesystem::path &path,
                       std::size_t slice_offset,
                       std::size_t slice_count) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open bank column file");
        }
        try {
            struct stat st {};

            if (::fstat(fd_, &st) || !S_ISREG(st.st_mode) || st.st_size < 256 ||
                static_cast<std::uintmax_t>(st.st_size) > SIZE_MAX) {
                throw std::runtime_error("invalid bank column file size/type");
            }
            bytes_ = static_cast<std::size_t>(st.st_size);
            mapping_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (mapping_ == MAP_FAILED) {
                mapping_ = nullptr;
                throw std::runtime_error("cannot map bank columns");
            }
            const auto *h = static_cast<const std::byte *>(mapping_);
            if (std::memcmp(h, "BJDATA01", 8) || u32(h + 8) != 1 || u32(h + 12) != 256) {
                throw std::invalid_argument("bank column header/version");
            }
            bank_ = u32(h + 16);
            n_ = u32(h + 20);
            const auto rows = u64(h + 24);
            bt_ = u32(h + 32);
            ba_ = u32(h + 36);
            if (n_ < 2 || n_ > 1024 || bank_ >= n_ || rows > (std::uint64_t{1} << 30) || !bt_ ||
                bt_ > 61 || !ba_ || ba_ > 62) {
                throw std::invalid_argument("bank column dimensions");
            }
            count_ = static_cast<std::size_t>(rows);
            std::size_t offset = 256;
            const std::array<unsigned, 5> widths{64, 1, 64, 96, 96};
            for (std::size_t j = 0; j < 5; ++j) {
                offset = (offset + 63) / 64 * 64;
                const auto *d = h + 64 + 24 * j;
                const auto size = bits_bytes(checked_bits(count_, widths[j]));
                if (u64(d) != offset || u64(d + 8) != size || u32(d + 16) != widths[j] ||
                    u32(d + 20) || offset > bytes_ || size > bytes_ - offset) {
                    throw std::invalid_argument("bank column directory/range");
                }
                columns_[j] = std::span(h + offset, size);
                offset += size;
            }
            if (offset != bytes_ ||
                (count_ % 8 && (std::to_integer<unsigned>(columns_[1].back()) >> (count_ % 8)))) {
                throw std::invalid_argument("bank column trailing data/padding");
            }
            if (slice_offset > count_ ||
                (slice_count != SIZE_MAX && slice_count > count_ - slice_offset)) {
                throw std::invalid_argument("bank slice outside file");
            }
            count_ = slice_count == SIZE_MAX ? count_ - slice_offset : slice_count;
            offset_ = slice_offset;
            for (std::size_t i = 0; i < count_; ++i) {
                if (!sid(i) || time(i) >= (std::uint64_t{1} << bt_) || !amount(i) ||
                    amount(i) >= (uint128_t{1} << ba_) || counterparty(i) >= (uint128_t{1} << 64)) {
                    throw std::invalid_argument("bank row outside declared domain");
                }
            }
        } catch (...) {
            if (mapping_) {
                ::munmap(mapping_, bytes_);
            }
            ::close(fd_);
            fd_ = -1;
            mapping_ = nullptr;
            throw;
        }
    }

    BankData::~BankData() {
        if (mapping_) {
            ::munmap(mapping_, bytes_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    bool BankData::fill(PreparedInput &in) const {
        if (in.spec.owner != bank_) {
            return false;
        }
        const auto suffix = "_" + std::to_string(bank_);
        const std::array<std::string, 5> names{
            "sid" + suffix, "dir" + suffix, "t" + suffix, "a" + suffix, "cp" + suffix};
        for (std::size_t j = 0; j < names.size(); ++j) {
            if (in.spec.name == names[j]) {
                const unsigned width = j == 1 ? 0u : j >= 3 ? 96u : 64u;
                if (in.spec.width != width || in.spec.count != count_) {
                    throw std::invalid_argument("bank input shape mismatch");
                }
                for (std::size_t i = 0; i < count_; ++i) {
                    if (!width) {
                        in.value_bits.set(i, outgoing(i));
                    } else if (j == 0) {
                        in.value64.data()[i] = sid(i);
                    } else if (j == 2) {
                        in.value64.data()[i] = time(i);
                    } else {
                        in.value128.data()[i] = j == 3 ? amount(i) : counterparty(i);
                    }
                }
                return true;
            }
        }
        return false;
    }
}
