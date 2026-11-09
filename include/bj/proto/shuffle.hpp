#pragma once
#include "bj/core/masked_material.hpp"
#include "bj/core/table.hpp"

namespace bj {
    class Shuffle {
        MaskedTable &table_;
        unsigned party_, parties_;
        std::vector<MaskedColumn> correction_, output_;
        AlignedBuffer<std::uint32_t> permutation_;
        AlignedBuffer<std::byte> received_, sent_;
        std::vector<std::size_t> offsets_;
        bool used_ = false;
        void check(Net &);
        void pack_input();
        void transform();
        void finish(std::span<const std::byte>);
        friend class ShuffleGroup;
        friend class ShuffleBatch;

    public:
        Shuffle(MaskedTable &, MaterialStore &, const MaterialPlan &, std::size_t index);
        void run(Net &);

        std::size_t bytes() const noexcept {
            return sent_.size();
        }
    };

    class ShuffleBatch {
        AlignedBuffer<std::byte> received_, sent_;

    public:
        explicit ShuffleBatch(std::size_t bytes) : received_(bytes), sent_(bytes) {
        }

        void run(Net &, std::span<Shuffle *const>);
    };

    class ShuffleGroup {
        std::vector<Shuffle *> shuffles_;
        AlignedBuffer<std::byte> received_, sent_;

    public:
        explicit ShuffleGroup(std::span<Shuffle *const>);
        void run(Net &);
    };
}
