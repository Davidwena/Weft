#include "allocation.hpp"
#include "bj/ops/quicksort.hpp"
#include "bj/proto/multiply.hpp"
#include <gtest/gtest.h>
#include <random>
#include <unistd.h>

namespace {
    using namespace bj;

    struct TemporaryDC {
        std::filesystem::path path;

        TemporaryDC() {
            char tmp[] = "/tmp/bj-t6-XXXXXX";
            const auto p = ::mkdtemp(tmp);
            if (!p) {
                throw std::runtime_error("mkdtemp");
            }
            path = p;
        }

        ~TemporaryDC() {
            std::error_code e;
            std::filesystem::remove_all(path, e);
        }
    };

    std::vector<std::byte>
    aggregate(unsigned k, std::size_t count, const std::vector<std::vector<std::byte>> &messages) {
        OpenLayout layout;
        if (k) {
            layout.add_ring(k, count);
        } else {
            layout.add_bits(count);
        }
        layout.seal();
        std::vector<std::byte> opened(layout.bytes());
        OpenBuf buf(layout, messages[0], opened);
        buf.initialize();
        for (unsigned party = 1; party < messages.size(); ++party) {
            for (std::size_t j = 0; j < layout.chunks().size(); ++j) {
                const auto &ch = layout.chunks()[j];
                buf.accumulate(j, std::span(messages[party]).subspan(ch.offset, ch.bytes));
            }
        }
        return opened;
    }

    template <class T> void check_pool(unsigned k, MaterialMode mode) {
        TemporaryDC tmp;
        MaterialPlan plan;
        plan.n = 3;
        plan.context = "DC pool unit";
        CompareSpec cmp;
        cmp.kind = CompareKind::MsbFresh;
        cmp.name = "pool";
        cmp.width = k;
        cmp.count = 151;
        plan.comparisons = {cmp};
        generate_materials(plan, mode, tmp.path / "m");
        std::array<std::unique_ptr<MaterialStore>, 3> stores;
        std::array<std::unique_ptr<FreshMsbPool<T>>, 3> pools;
        for (unsigned party = 0; party < 3; ++party) {
            stores[party] = std::make_unique<MaterialStore>(
                plan, party, tmp.path / "m" / ("party_" + std::to_string(party) + ".bin"));
            stores[party]->preload();
            pools[party] = std::make_unique<FreshMsbPool<T>>(*stores[party], plan, 0, 79);
        }
        std::mt19937_64 rng(6831);
        std::size_t used = 0;
        for (auto count : {1u, 65u, 79u, 6u}) {
            const auto inputs = pools[0]->inputs();
            std::vector<std::vector<T>> share(3, std::vector<T>(count));
            std::vector<std::int64_t> plain(count);
            for (unsigned i = 0; i < count; ++i) {
                plain[i] = static_cast<std::int64_t>(rng() % 2001) - 1000;
                share[0][i] = (T(rng()) << 32) ^ rng();
                share[1][i] = (T(rng()) << 32) ^ rng();
                share[2][i] = T(plain[i]) - share[0][i] - share[1][i];
            }
            std::vector<std::vector<std::byte>> messages(
                3, std::vector<std::byte>(bits_bytes(count * k)));
            for (unsigned party = 0; party < 3; ++party) {
                pools[party]->reserve(count);
                pools[party]->opening(share[party], messages[party]);
            }
            auto opened = aggregate(k, count, messages);
            for (auto &bytes : messages) {
                bytes.assign(bits_bytes(count * inputs), std::byte{0});
            }
            for (unsigned party = 0; party < 3; ++party) {
                pools[party]->table_opening(opened, messages[party]);
            }
            opened = aggregate(0, count * inputs, messages);
            for (auto &bytes : messages) {
                bytes.assign(bits_bytes(count), std::byte{0});
            }
            for (unsigned party = 0; party < 3; ++party) {
                pools[party]->reveal(opened, messages[party]);
            }
            opened = aggregate(0, count, messages);
            for (auto &pool : pools) {
                pool->finish(opened);
            }
            for (unsigned i = 0; i < count; ++i) {
                EXPECT_EQ(pools[0]->less(i), plain[i] < 0);
            }
            used += count;
            EXPECT_EQ(pools[0]->used(), used);
        }
        EXPECT_THROW(pools[0]->reserve(1), std::runtime_error);
        EXPECT_THROW(stores[0]->claim_comparison(0, 0), std::logic_error);
    }

    TEST(DCPool, Fresh64And72NonalignedSlicesAndExhaustion) {
        check_pool<std::uint64_t>(64, MaterialMode::Random);
        check_pool<uint128_t>(72, MaterialMode::Prg);
    }
}
