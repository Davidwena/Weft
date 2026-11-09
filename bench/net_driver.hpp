#pragma once
#include "bj/common/aligned.hpp"
#include "bj/common/config.hpp"
#include "bj/common/ring.hpp"
#include "bj/common/timer.hpp"
#include "bj/net/net.hpp"
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <string_view>

namespace bj {
    struct NetOptions {
        unsigned id = UINT32_MAX, n = 0, iterations = 5, warmup = 2, king = UINT32_MAX;
        std::size_t bytes = 64 * 1024 * 1024;
        std::string hosts, mode = "open", open_mode, kind = "ring64";
    };

    inline NetOptions net_options(int argc, char **argv) {
        NetOptions o;
        for (int i = 1; i < argc; ++i) {
            std::string_view key = argv[i], value;
            if (auto eq = key.find('='); eq != std::string_view::npos) {
                value = key.substr(eq + 1);
                key = key.substr(0, eq);
            } else {
                if (i + 1 == argc) {
                    throw std::invalid_argument("every option requires a value");
                }
                value = argv[++i];
            }
            if (key == "--hosts") {
                o.hosts = value;
            } else if (key == "--mode") {
                o.mode = value;
            } else if (key == "--open-mode") {
                parse_open_mode(std::string(value));
                o.open_mode = value;
            } else if (key == "--kind") {
                o.kind = value;
            } else {
                std::uint64_t x = 0;
                auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), x);
                if (ec != std::errc{} || end != value.data() + value.size()) {
                    throw std::invalid_argument("invalid numeric option");
                }
                if (key == "--bytes") {
                    if (x > SIZE_MAX) {
                        throw std::invalid_argument("bytes too large");
                    }
                    o.bytes = x;
                } else {
                    if (x > UINT32_MAX) {
                        throw std::invalid_argument("option too large");
                    }
                    if (key == "--id") {
                        o.id = static_cast<unsigned>(x);
                    } else if (key == "--n") {
                        o.n = static_cast<unsigned>(x);
                    } else if (key == "--king") {
                        o.king = static_cast<unsigned>(x);
                    } else if (key == "--iterations") {
                        o.iterations = static_cast<unsigned>(x);
                    } else if (key == "--warmup") {
                        o.warmup = static_cast<unsigned>(x);
                    } else {
                        throw std::invalid_argument("unknown option");
                    }
                }
            }
        }
        if (o.hosts.empty() || o.n < 2 || o.id >= o.n || !o.iterations ||
            (o.mode != "open" && o.mode != "exchange" && o.mode != "pingpong") ||
            (o.mode == "pingpong" && o.n != 2)) {
            throw std::invalid_argument(
                "invalid options: --id i --n n --hosts file [--mode open|exchange|pingpong --open-mode=auto|broadcast --king k --kind bits|ring22|ring32|ring49|ring64|mixed --bytes m]");
        }
        return o;
    }

    inline std::uint64_t sample_word(unsigned id, std::size_t i) noexcept {
        return (UINT64_C(0x9e3779b97f4a7c15) * (i + 1)) ^ (UINT64_MAX - id * 97);
    }

    inline OpenLayout bench_layout(const NetOptions &o) {
        OpenLayout l;
        if (o.bytes > (SIZE_MAX - 7) / 8) {
            throw std::length_error("benchmark size overflow");
        }
        if (o.kind == "bits") {
            l.add_bits(o.bytes * 8);
        } else if (o.kind == "mixed") {
            const auto count = o.bytes / 32;
            l.add_ring(22, count + 1)
                .add_ring(32, count + 3)
                .add_ring(49, count + 5)
                .add_ring(64, count + 7)
                .add_bits(o.bytes + 3);
        } else {
            unsigned k = 0;
            if (o.kind == "ring22") {
                k = 22;
            } else if (o.kind == "ring32") {
                k = 32;
            } else if (o.kind == "ring49") {
                k = 49;
            } else if (o.kind == "ring64") {
                k = 64;
            } else {
                throw std::invalid_argument("invalid benchmark kind");
            }
            if (o.bytes * 8 % k) {
                throw std::invalid_argument("bytes must encode a whole number of ring elements");
            }
            l.add_ring(k, o.bytes * 8 / k);
        }
        l.seal();
        return l;
    }

    template <unsigned K>
    void fill_field(std::span<std::byte> out, std::size_t count, unsigned id) {
        using T = typename Ring<K>::T;
        std::vector<T> values(count);
        for (std::size_t i = 0; i < count; ++i) {
            values[i] = Ring<K>::normalize(static_cast<T>(sample_word(id, i)));
        }
        pack_ring<K>(values, out);
    }

    template <unsigned K>
    bool verify_field(std::span<const std::byte> in, std::size_t count, unsigned n) {
        using T = typename Ring<K>::T;
        std::vector<T> values(count);
        unpack_ring<K>(in, values);
        for (std::size_t i = 0; i < count; ++i) {
            T expected = 0;
            for (unsigned p = 0; p < n; ++p) {
                expected = Ring<K>::add(expected, static_cast<T>(sample_word(p, i)));
            }
            if (values[i] != expected) {
                return false;
            }
        }
        return true;
    }

    inline nlohmann::json network_run(const NetOptions &o) {
        auto c = Config::load(o.hosts);
        if (!o.open_mode.empty()) {
            c.open_mode = parse_open_mode(o.open_mode);
        }
        if (o.king != UINT32_MAX) {
            c.king = o.king;
        }
        c.validate(o.n);
        omp_set_num_threads(static_cast<int>(c.threads));
        Net net(o.id, c);
        OpenLayout layout;
        if (o.mode == "open") {
            layout = bench_layout(o);
        } else {
            layout.add_bits(o.bytes * 8);
            layout.seal();
        }
        AlignedBuffer<std::byte> share(layout.bytes()), opened(layout.bytes());
        OpenBuf buf(layout, share.span(), opened.span());
        if (o.mode == "open") {
            for (const auto &f : layout.fields()) {
                auto out = share.span().subspan(f.offset, f.bytes);
                if (f.width == 22) {
                    fill_field<22>(out, f.count, o.id);
                } else if (f.width == 32) {
                    fill_field<32>(out, f.count, o.id);
                } else if (f.width == 49) {
                    fill_field<49>(out, f.count, o.id);
                } else if (f.width == 64) {
                    fill_field<64>(out, f.count, o.id);
                } else {
                    for (std::size_t i = 0; i < out.size(); ++i) {
                        out[i] = std::byte((i * 31 + o.id * 17) & 255);
                    }
                }
            }
        } else {
            std::fill(share.span().begin(), share.span().end(), std::byte(o.id + 1));
        }

        BufferPool raw(o.mode == "exchange" ? o.n : 0, o.bytes);
        std::vector<std::span<const std::byte>> tx(o.n);
        std::vector<std::span<std::byte>> rx(o.n);
        for (unsigned p = 0; p < o.n; ++p) {
            if (p != o.id) {
                tx[p] = share.span();
                if (o.mode == "exchange") {
                    rx[p] = raw.get(p, o.bytes);
                }
            }
        }
        auto step = [&] {
            if (o.mode == "open") {
                net.open_round(buf);
            } else if (o.mode == "exchange") {
                net.exchange_all(tx, rx);
            } else if (o.id == 0) {
                net.send(1, share.span());
                net.recv(1, opened.span());
            } else {
                net.recv(0, opened.span());
                net.send(0, share.span());
            }
        };
        for (unsigned r = 0; r < o.warmup; ++r) {
            step();
        }

        std::vector<std::span<const std::byte>> empty_tx(o.n);
        std::vector<std::span<std::byte>> empty_rx(o.n);
        net.exchange_all(empty_tx, empty_rx);
        net.reset_stats();
        std::vector<double> times(o.iterations);
        Timer total;
        for (unsigned r = 0; r < o.iterations; ++r) {
            Timer t;
            step();
            times[r] = t.seconds();
        }
        const auto elapsed = total.seconds();
        bool correct = true;
        if (o.mode == "open") {
            for (const auto &f : layout.fields()) {
                auto in = opened.span().subspan(f.offset, f.bytes);
                if (f.width == 22) {
                    correct &= verify_field<22>(in, f.count, o.n);
                } else if (f.width == 32) {
                    correct &= verify_field<32>(in, f.count, o.n);
                } else if (f.width == 49) {
                    correct &= verify_field<49>(in, f.count, o.n);
                } else if (f.width == 64) {
                    correct &= verify_field<64>(in, f.count, o.n);
                } else {
                    for (std::size_t i = 0; i < in.size(); ++i) {
                        unsigned expected = 0;
                        for (unsigned p = 0; p < o.n; ++p) {
                            expected ^= (i * 31 + p * 17) & 255;
                        }
                        if (i + 1 == in.size() && f.count % 8) {
                            expected &= (1u << (f.count % 8)) - 1;
                        }
                        correct &= (in[i] == std::byte(expected));
                    }
                }
            }
        } else if (o.mode == "exchange") {
            for (unsigned p = 0; p < o.n; ++p) {
                if (p != o.id) {
                    correct &= std::all_of(rx[p].begin(), rx[p].end(), [&](auto v) {
                        return v == std::byte(p + 1);
                    });
                }
            }
        } else {
            correct = std::all_of(opened.span().begin(), opened.span().end(), [&](auto v) {
                return v == std::byte(2 - o.id);
            });
        }
        if (!correct) {
            throw std::runtime_error("opened value/payload differs from plaintext reference");
        }
        nlohmann::json peers = nlohmann::json::array();
        for (unsigned p = 0; p < o.n; ++p) {
            const auto &s = net.stats()[p];
            peers.push_back({{"peer", p},
                             {"sent_payload", s.sent_payload},
                             {"received_payload", s.received_payload},
                             {"sent_wire", s.sent_wire},
                             {"received_wire", s.received_wire},
                             {"sent_messages", s.sent_messages},
                             {"received_messages", s.received_messages}});
        }
        const auto &opens = net.open_stats();
        return {{"id", o.id},
                {"n", o.n},
                {"share_bytes", layout.bytes()},
                {"share_bits", layout.bits()},
                {"mode", o.mode},
                {"kind", o.kind},
                {"open_mode", open_mode_name(c.open_mode)},
                {"effective_open_mode", net.uses_king() ? "king" : "broadcast"},
                {"king", c.king},
                {"iterations", o.iterations},
                {"warmup", o.warmup},
                {"seconds", elapsed},
                {"iteration_seconds", times},
                {"exchange_rounds", net.rounds()},
                {"open_rounds", opens.rounds},
                {"network_hops", opens.hops},
                {"contribution_bits", opens.contribution_bits},
                {"reduced_chunks", opens.reduced_chunks},
                {"chunks_before_all_received", opens.chunks_before_all_received},
                {"receive_scratch_bytes", net.receive_scratch_bytes()},
                {"threads", c.threads},
                {"peers", peers},
                {"payload_verified", true}};
    }

    inline int net_entry(int argc, char **argv) {
        try {
            const auto j = network_run(net_options(argc, argv));
            std::puts(j.dump().c_str());
            return 0;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "network: %s\n", e.what());
            return 1;
        }
    }
}
