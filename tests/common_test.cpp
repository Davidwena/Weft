#include "bj/common/ring.hpp"
#include "bj/common/bitvec.hpp"
#include "bj/common/prg.hpp"
#include "bj/common/timer.hpp"
#include "bj/common/config.hpp"
#include "allocation.hpp"
#include <gtest/gtest.h>
#include <array>
#include <fstream>
#include <limits>
#include <omp.h>
#include <unistd.h>

namespace {
    using namespace bj;

    std::array<std::byte, 16> hex(const char *s) {
        std::array<std::byte, 16> a{};
        for (unsigned i = 0; i < 16; ++i) {
            unsigned x;
            std::sscanf(s + 2 * i, "%2x", &x);
            a[i] = std::byte(x);
        }
        return a;
    }

    TEST(Ring, ExhaustiveSmall) {
        for (unsigned x = 0; x < 256; ++x) {
            for (unsigned y = 0; y < 256; ++y) {
                ASSERT_EQ(Ring<8>::add(x, y), (x + y) % 256);
                ASSERT_EQ(Ring<8>::sub(x, y), (x - y) & 255);
                ASSERT_EQ(Ring<8>::mul(x, y), (x * y) % 256);
                ASSERT_EQ(Ring<8>::msb(x), x >= 128);
            }
        }
    }

    template <unsigned K> void roundtrip() {
        using R = Ring<K>;
        using T = typename R::T;
        std::array<T, 137> a{}, b{};
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = R::normalize(static_cast<T>(0x9e3779b97f4a7c15ULL * i));
        }
        a[0] = 0;
        a[1] = R::mask();
        a[2] = T{1} << (K - 1);
        std::array<std::byte, packed_ring_bytes<K>(137)> packed{};
        pack_ring<K>(a, packed);
        unpack_ring<K>(packed, b);
        EXPECT_EQ(a, b);
        if constexpr ((K * 137) % 8) {
            EXPECT_EQ(std::to_integer<unsigned>(packed.back()) >> ((K * 137) % 8), 0);
        }
    }

    TEST(Ring, WidthsAndPacking) {
        static_assert(std::is_same_v<Ring<22>::T, std::uint32_t>);
        static_assert(std::is_same_v<Ring<49>::T, std::uint64_t>);
        static_assert(Ring<64>::mask() == UINT64_MAX);
        EXPECT_EQ(Ring<64>::add(UINT64_MAX, 1), 0);
        EXPECT_EQ(Ring<64>::sub(0, 1), UINT64_MAX);
        EXPECT_EQ(Ring<32>::add(UINT32_MAX, 1), 0);
        EXPECT_EQ(Ring<64>::reduce<22>(UINT64_MAX), Ring<22>::mask());
        roundtrip<1>();
        roundtrip<7>();
        roundtrip<22>();
        roundtrip<32>();
        roundtrip<49>();
        roundtrip<64>();
        EXPECT_THROW(packed_ring_bytes<64>(SIZE_MAX), std::length_error);
    }

    TEST(BitVec, BoundariesAndAliasing) {
        for (std::size_t n : {0, 1, 7, 8, 63, 64, 65, 129, 1025}) {
            BitVec a(n), b(n), out(n), copy(n);
            AlignedBuffer<std::byte> bytes(a.packed_bytes());
            for (std::size_t i = 0; i < n; ++i) {
                a.set(i, i % 3 == 0);
                b.set(i, i % 5 == 0);
            }
            BitVec::xor_into(a, b, out);
            for (std::size_t i = 0; i < n; ++i) {
                ASSERT_EQ(out.get(i), (i % 3 == 0) ^ (i % 5 == 0));
            }
            out.pack(bytes.span());
            copy.unpack(bytes.span());
            for (std::size_t i = 0; i < n; ++i) {
                ASSERT_EQ(copy.get(i), out.get(i));
            }
            BitVec::and_into(a, b, a);
            for (std::size_t i = 0; i < n; ++i) {
                ASSERT_EQ(a.get(i), i % 15 == 0);
            }
            if (n % 8) {
                std::fill(bytes.span().begin(), bytes.span().end(), std::byte{255});
                copy.unpack(bytes.span());
                copy.pack(bytes.span());
                EXPECT_EQ(std::to_integer<unsigned>(bytes.span().back()), (1u << (n % 8)) - 1);
            }
            EXPECT_THROW(a.get(n), std::out_of_range);
        }
    }

    TEST(Storage, AlignedAndPool) {
        BufferPool pool(3, 4096);
        auto a = pool.get(0, 512), b = pool.get(0, 512);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % 64, 0);
        EXPECT_EQ(a.data(), b.data());
        EXPECT_THROW(pool.get(0, 4097), std::length_error);
        EXPECT_THROW(pool.get(3, 1), std::out_of_range);
    }

    TEST(Prg, Fips197AndNistCounter) {
        Prg p(hex("000102030405060708090a0b0c0d0e0f"));
        auto input = hex("00112233445566778899aabbccddeeff");
        std::array<std::byte, 16> result{};
        p.encrypt_block(input, result);
        EXPECT_EQ(result, hex("69c4e0d86a7b0430d8cdb78070b4c55a"));
        Prg nist(hex("2b7e151628aed2a6abf7158809cf4f3c"));
        nist.encrypt_block(hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"), result);
        EXPECT_EQ(result, hex("ec8cdf7398607cb0f2d21675ea9ea1e4"));
        Prg zero(Prg::Key{});
        std::array<std::byte, 32> ctr{};
        zero.fill(0, 0, ctr);
        EXPECT_TRUE(std::equal(
            ctr.begin(), ctr.begin() + 16, hex("66e94bd4ef8a2c3b884cfa59ca342b2e").begin()));
        EXPECT_TRUE(std::equal(
            ctr.begin() + 16, ctr.end(), hex("58e2fccefa7e3061367f1d57a4e7455a").begin()));
    }

    TEST(Prg, RandomAccessAndDomainSeparation) {
        Prg p(hex("000102030405060708090a0b0c0d0e0f"));
        std::array<std::byte, 1027> full{}, joined{};
        p.fill(123, 0, full);
        for (std::size_t offset = 0; offset < full.size(); offset += 13) {
            auto piece =
                std::span(joined).subspan(offset, std::min<std::size_t>(13, full.size() - offset));
            p.fill(123, offset, piece);
        }
        EXPECT_EQ(full, joined);
        p.fill(124, 0, joined);
        EXPECT_NE(full, joined);

        auto counter = hex("000000000000007b0000000000000007");
        std::array<std::byte, 16> block{};
        p.encrypt_block(counter, block);
        EXPECT_TRUE(std::equal(block.begin(), block.end(), full.begin() + 112));
        EXPECT_THROW(p.fill(0, UINT64_MAX, std::span(joined).first(2)), std::length_error);
    }

    TEST(Common, NoHotAllocations) {
        omp_set_num_threads(2);
        BitVec a(1 << 21), b(1 << 21), out(1 << 21);
        BufferPool pool(2, 8192);
        Prg p(Prg::Key{});
        std::array<std::byte, 8192> bytes{};
        BitVec::xor_into(a, b, out);
        BitVec::and_into(a, b, out);
        test::allocations = 0;
        test::count_allocations = true;
        for (unsigned i = 0; i < 50; ++i) {
            p.fill(1, i * bytes.size(), bytes);
            BitVec::xor_into(a, b, out);
            BitVec::and_into(a, b, out);
            (void)pool.get(0, 8192);
        }
        test::count_allocations = false;
        EXPECT_EQ(test::allocations.load(), 0);
    }

    TEST(Config, LoadAndReject) {
        const auto path = "/tmp/bj-config-" + std::to_string(::getpid()) + ".json";
        {
            std::ofstream f(path);
            f << R"({"hosts":[{"host":"127.0.0.1","port":19001},{"host":"127.0.0.1","port":19002}],"threads":2,"king":1,"open_mode":"broadcast"})";
        }
        const auto c = Config::load(path);
        EXPECT_EQ(c.threads, 2);
        EXPECT_EQ(c.hosts.size(), 2);
        EXPECT_EQ(c.king, 1);
        EXPECT_EQ(c.open_mode, OpenMode::Broadcast);
        EXPECT_THROW(parse_open_mode("king"), std::invalid_argument);
        auto bad_king = c;
        bad_king.king = 2;
        EXPECT_THROW(bad_king.validate(2), std::invalid_argument);
        EXPECT_THROW(c.validate(3), std::invalid_argument);
        {
            std::ofstream f(path);
            f << R"({"hosts":[{"host":"x","port":-1}]})";
        }
        EXPECT_THROW(Config::load(path), std::invalid_argument);
        ::unlink(path.c_str());
    }

    TEST(Timer, Phases) {
        Timer t;
        PhaseStats s;
        s.add(Phase::Input, t.seconds(), 12, 24);
        s.add(Phase::Input, 0.1, 2, 3);
        EXPECT_GE(s[Phase::Input].seconds, 0.1);
        EXPECT_EQ(s[Phase::Input].sent, 14);
        EXPECT_EQ(s[Phase::Online].sent, 0);
    }
}
