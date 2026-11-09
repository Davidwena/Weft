#pragma once
#include "bj/core/material.hpp"
#include "bj/common/bitvec.hpp"
#include "bj/net/net.hpp"

namespace bj {
    struct PreparedInput {
        InputSpec spec;
        std::size_t wire_offset = 0;
        BitVec mask_bits, delta_bits, share_bits, owner_bits, value_bits;
        AlignedBuffer<std::uint32_t> mask32, delta32, share32, owner32, value32;
        AlignedBuffer<std::uint64_t> mask64, delta64, share64, owner64, value64;
        AlignedBuffer<uint128_t> mask128, delta128, share128, owner128, value128;
        PreparedInput(const InputSpec &input, unsigned party);
    };

    class InputManager {
        unsigned party_, n_;
        std::vector<std::unique_ptr<PreparedInput>> inputs_;
        std::vector<AlignedBuffer<std::byte>> wire_;
        std::vector<std::span<const std::byte>> send_;
        std::vector<std::span<std::byte>> recv_;
        bool used_ = false;

    public:
        InputManager(MaterialStore &material,
                     const MaterialPlan &plan,
                     const std::filesystem::path &private_input,
                     const std::filesystem::path &bank_file = {},
                     std::size_t bank_offset = 0,
                     std::size_t bank_count = SIZE_MAX);
        void run(Net &net);

        const PreparedInput &at(std::size_t i) const {
            return *inputs_.at(i);
        }

        std::span<const std::uint64_t> bits(std::size_t i) const {
            const auto &in = at(i);
            if (in.spec.width) {
                throw std::invalid_argument("expected Boolean input");
            }
            return in.share_bits.words();
        }

        std::size_t owner_wire_bytes(unsigned owner) const {
            return wire_.at(owner).size();
        }
    };
}
