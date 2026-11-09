#include "../dealer/parallel.hpp"
#include <iostream>
#include <numeric>
using namespace bj;

void equal(std::span<const std::byte> a, std::span<const std::byte> b) {
    if (a.size() != b.size() || !std::equal(a.begin(), a.end(), b.begin())) {
        throw std::runtime_error("parallel kernel differs from original");
    }
}

int main() {
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(16);
        Prg prg(Prg::Key{});
        std::size_t checks = 0;
        for (std::size_t n : {131073ul, 131074ul, 131080ul}) {
            for (unsigned k : {1u, 22u, 26u, 32u, 49u, 64u, 72u, 96u}) {
                AlignedBuffer<uint128_t> native(n), decoded(n);
                AlignedBuffer<std::byte> a(bits_bytes(n * k)), b(a.size());
                for (std::size_t i = 0; i < n; ++i) {
                    native.data()[i] =
                        (uint128_t(i) * 1234567890123456789ULL) & native_mask<uint128_t>(k);
                }
                pack_native<uint128_t>(k, native.span(), a.span());
                x11_pack<uint128_t>(k, native.span(), b.span());
                equal(a.span(), b.span());
                x11_unpack<uint128_t>(k, b.span(), decoded.span());
                if (!std::equal(
                        native.span().begin(), native.span().end(), decoded.span().begin())) {
                    throw std::runtime_error("parallel unpack differs");
                }
                ++checks;
            }
            for (auto shape : {CompareShape(64, 7, true),
                               CompareShape(25, 5, false),
                               CompareShape(32, 4, true)}) {
                AlignedBuffer<std::byte> a(bits_bytes(shape.eval_bits() * n)), b(a.size());
                auto random = [&](auto out) {
                    prg.fill(17, 0, out);
                };
                make_comparison_evaluation(shape, n, a.span(), random);
                x11_evaluation(shape, n, b.span(), random);
                equal(a.span(), b.span());
                AlignedBuffer<std::uint64_t> thresholds(n);
                for (std::size_t i = 0; i < n; ++i) {
                    thresholds.data()[i] = i * 733;
                }
                AlignedBuffer<std::byte> t(bits_bytes(shape.table_bits() * n)), u(t.size());
                make_comparison_table<std::uint64_t>(shape, thresholds.span(), t.span());
                x11_table<std::uint64_t>(shape, thresholds.span(), u.span());
                equal(t.span(), u.span());
                ++checks;
            }
            for (unsigned width : {0u, 32u, 64u, 96u}) {
                auto k = width ? width : 1;
                AlignedBuffer<std::uint32_t> permutation(n);
                for (std::size_t i = 0; i < n; ++i) {
                    permutation.data()[i] = static_cast<std::uint32_t>(n - i - 1);
                }
                AlignedBuffer<std::byte> a(bits_bytes(k * n)), b(a.size()), c(a.size()),
                    d(a.size());
                prg.fill(21, 0, a.span());
                prg.fill(22, 0, b.span());
                std::fill(c.span().begin(), c.span().end(), std::byte{0});
                std::fill(d.span().begin(), d.span().end(), std::byte{0});
                for (std::size_t i = 0; i < n; ++i) {
                    auto x = load_wide(a.span(), i * k, k),
                         y = load_wide(b.span(), std::size_t(permutation.data()[i]) * k, k);
                    store_wide(
                        c.span(), i * k, k, width ? (x - y) & native_mask<uint128_t>(k) : x ^ y);
                }
                x11_correction(width, n, permutation.span(), a.span(), b.span(), d.span());
                equal(c.span(), d.span());
                ++checks;
            }
            for (unsigned k : {1u, 49u, 64u, 96u}) {
                MaskSpec spec;
                spec.width = k;
                MaskTerm term{0, 0, k == 49 ? 0ul : 3ul, n, 1, n, -7};
                AlignedBuffer<std::byte> source(bits_bytes(n * k)), a(bits_bytes((n + 8) * k)),
                    b(a.size());
                prg.fill(23, 0, source.span());
                prg.fill(24, 0, a.span());
                std::copy(a.span().begin(), a.span().end(), b.span().begin());
                for (std::size_t j = 0; j < n; ++j) {
                    auto i = term.destination_offset + j;
                    auto x = load_wide(source.span(), j * k, k),
                         old = load_wide(a.span(), i * k, k);
                    store_wide(a.span(),
                               i * k,
                               k,
                               (old + static_cast<uint128_t>(term.coefficient) * x) &
                                   native_mask<uint128_t>(k));
                }
                x11_mask_term(spec, term, source.span(), k, b.span());
                equal(a.span(), b.span());
                ++checks;
            }
        }
        AlignedBuffer<std::byte> a(3 * (1 << 20) + 13), b(a.size());
        prg.fill(98, 0, a.span());
        x11_prg(prg, 98, b.span());
        equal(a.span(), b.span());
        ++checks;
        std::cout << "{\"passed\":true,\"bit_exact_kernel_checks\":" << checks
                  << ",\"workers\":16,\"includes_non_byte_aligned_counts\":true}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
