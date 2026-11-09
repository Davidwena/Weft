#pragma once
#include "bj/common/aligned.hpp"
#include "bj/net/net.hpp"
#include "bj/proto/shuffle.hpp"
#include <functional>

namespace bj {

    struct NetworkHook {
        OpenStats counters{};
        virtual ~NetworkHook() = default;
        virtual void open(OpenBuf &) = 0;
        virtual void shuffle(std::span<Shuffle *const>) = 0;
        virtual void wait_shared(std::size_t) = 0;
    };

    struct OpenAugment {
        virtual ~OpenAugment() = default;
        virtual std::span<OpenBuf *const> prepare() = 0;
        virtual void consume() = 0;
    };

    struct MergeStats {
        std::uint64_t unmerged_opens = 0, merged_opens = 0, unmerged_shuffle_hops = 0,
                      merged_shuffle_hops = 0;
        std::uint64_t graph_rounds = 0, contribution_bits = 0, shuffle_bytes = 0;
        std::vector<std::uint64_t> wave_rounds;
    };

    struct ScheduledJob {
        std::function<void()> run;
        std::vector<std::size_t> dependencies;
    };

    class MultiRound {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        MultiRound(Net &,
                   std::vector<ScheduledJob>,
                   std::size_t max_bytes,
                   std::size_t max_fields,
                   unsigned threads,
                   bool merge = true,
                   OpenAugment *augment = nullptr);
        ~MultiRound();
        void run();
        const MergeStats &stats() const;
    };
}
