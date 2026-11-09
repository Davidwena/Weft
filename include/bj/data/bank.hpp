#pragma once
#include "bj/common/packed.hpp"
#include <array>
#include <filesystem>

namespace bj {
    struct PreparedInput;

    class BankData {
        int fd_ = -1;
        void *mapping_ = nullptr;
        std::size_t bytes_ = 0;
        std::array<std::span<const std::byte>, 5> columns_{};
        unsigned bank_ = 0, n_ = 0, bt_ = 0, ba_ = 0;
        std::size_t count_ = 0, offset_ = 0;

    public:
        explicit BankData(const std::filesystem::path &,
                          std::size_t offset = 0,
                          std::size_t count = SIZE_MAX);
        ~BankData();
        BankData(const BankData &) = delete;
        BankData &operator=(const BankData &) = delete;

        unsigned bank() const {
            return bank_;
        }

        unsigned parties() const {
            return n_;
        }

        unsigned time_bits() const {
            return bt_;
        }

        unsigned amount_bits() const {
            return ba_;
        }

        std::size_t size() const {
            return count_;
        }

        std::uint64_t sid(std::size_t i) const {
            return load_packed(columns_[0], (i + offset_) * 64, 64);
        }

        bool outgoing(std::size_t i) const {
            return load_packed(columns_[1], i + offset_, 1) != 0;
        }

        std::uint64_t time(std::size_t i) const {
            return load_packed(columns_[2], (i + offset_) * 64, 64);
        }

        uint128_t amount(std::size_t i) const {
            return load_wide(columns_[3], (i + offset_) * 96, 96);
        }

        uint128_t counterparty(std::size_t i) const {
            return load_wide(columns_[4], (i + offset_) * 96, 96);
        }

        bool fill(PreparedInput &) const;
    };
}
