#include "bj/core/party.hpp"
#include "bj/core/input.hpp"
#include "bj/core/round_exec.hpp"
#include "bj/proto/b2a.hpp"
#include "bj/proto/and2a.hpp"
#include "bj/common/config.hpp"
#include "bj/common/timer.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <sys/resource.h>

namespace bj {
    namespace {
        using Json = nlohmann::json;

        std::string decimal_wide(uint128_t value) {
            std::string result;
            do {
                result.push_back(static_cast<char>('0' + value % 10));
                value /= 10;
            } while (value);
            std::reverse(result.begin(), result.end());
            return result;
        }

        struct PreparedGate {
            virtual ~PreparedGate() = default;
            virtual Step &step() = 0;
            virtual void output_share(std::span<std::byte> wire, bool sum) = 0;
            virtual Json benchmark(unsigned repeats, unsigned threads) = 0;
        };

        template <unsigned M, class T> struct NativeGate final : PreparedGate {
            const GateSpec &spec;
            AlignedBuffer<T> output;
            FusedAnd2A<M, T> protocol;

            static std::array<std::span<const std::uint64_t>, M> views(const GateSpec &s,
                                                                       const InputManager &inputs) {
                if (s.inputs.size() != M) {
                    throw std::invalid_argument("gate inputs require workflow execution");
                }
                std::array<std::span<const std::uint64_t>, M> value;
                for (unsigned i = 0; i < M; ++i) {
                    value[i] = inputs.bits(s.inputs[i]);
                }
                return value;
            }

            NativeGate(MaterialStore &store,
                       const MaterialPlan &plan,
                       std::size_t index,
                       const InputManager &inputs)
                : spec(plan.gates[index]), output(spec.count),
                  protocol(store, plan, index, views(spec, inputs), output.span()) {
            }

            Step &step() override {
                return protocol;
            }

            void output_share(std::span<std::byte> wire, bool sum) override {
                if (!sum) {
                    pack_native<T>(spec.width, output.span(), wire);
                    return;
                }
                T value = 0;
#pragma omp parallel for if (output.size() >= 16384) reduction(+ : value) schedule(static)
                for (std::size_t i = 0; i < output.size(); ++i) {
                    value += output.data()[i];
                }
                value &= native_mask<T>(spec.width);
                pack_native<T>(spec.width, std::span<const T>(&value, 1), wire);
            }

            Json benchmark(unsigned repeats, unsigned threads) override {
                Json rows = Json::array();
                if (!repeats) {
                    return rows;
                }
                std::array<std::span<const std::uint64_t>, M> hats;
                for (unsigned i = 0; i < M; ++i) {
                    hats[i] = protocol.opened_column(i);
                }
                for (unsigned count : {1u, threads}) {
                    omp_set_num_threads(static_cast<int>(count));
                    evaluate_and2a<M, T>(protocol.local_table(), hats, output.span());
                    Timer t;
                    for (unsigned r = 0; r < repeats; ++r) {
                        evaluate_and2a<M, T>(protocol.local_table(), hats, output.span());
                    }
                    const auto seconds = t.seconds();
                    std::uint64_t check = 0;
                    for (std::size_t i = 0; i < output.size(); i += 4096) {
                        check ^= static_cast<std::uint64_t>(output.data()[i]);
                    }
                    asm volatile("" : : "r"(check) : "memory");
                    rows.push_back(
                        {{"threads", count},
                         {"repeats", repeats},
                         {"seconds", seconds},
                         {"instances_per_second",
                          seconds ? static_cast<double>(spec.count) * repeats / seconds : 0}});
                }
                omp_set_num_threads(static_cast<int>(threads));
                return rows;
            }
        };

        std::unique_ptr<PreparedGate> prepare_gate(MaterialStore &store,
                                                   const MaterialPlan &plan,
                                                   std::size_t index,
                                                   const InputManager &inputs) {
            const auto &s = plan.gates[index];
            auto make = [&]<class T>() -> std::unique_ptr<PreparedGate> {
                switch (s.m) {
                case 1:
                    return std::make_unique<NativeGate<1, T>>(store, plan, index, inputs);
                case 2:
                    return std::make_unique<NativeGate<2, T>>(store, plan, index, inputs);
                case 3:
                    return std::make_unique<NativeGate<3, T>>(store, plan, index, inputs);
                case 4:
                    return std::make_unique<NativeGate<4, T>>(store, plan, index, inputs);
                default:
                    throw std::invalid_argument("invalid AND arity");
                }
            };
            if (s.width <= 32) {
                return make.template operator()<std::uint32_t>();
            }
            if (s.width <= 64) {
                return make.template operator()<std::uint64_t>();
            }
            return make.template operator()<uint128_t>();
        }

        Json network_stats(const Net &net) {
            Json peers = Json::array();
            for (unsigned p = 0; p < net.size(); ++p) {
                const auto &s = net.stats()[p];
                peers.push_back({{"peer", p},
                                 {"sent_payload", s.sent_payload},
                                 {"received_payload", s.received_payload},
                                 {"sent_wire", s.sent_wire},
                                 {"received_wire", s.received_wire},
                                 {"sent_messages", s.sent_messages},
                                 {"received_messages", s.received_messages}});
            }
            return {{"peers", peers},
                    {"open_rounds", net.open_stats().rounds},
                    {"network_hops", net.open_stats().hops},
                    {"contribution_bits", net.open_stats().contribution_bits},
                    {"raw_exchange_rounds", net.rounds()}};
        }

        void check_session(Net &net, MaterialStore &store, const MaterialPlan &plan) {
            std::array<std::byte, 24> context{};
            std::memcpy(context.data(), store.run_id().data(), 16);
            const auto hash = plan.fingerprint();
            std::memcpy(context.data() + 16, &hash, 8);
            std::vector<std::array<std::byte, 24>> from(net.size());
            std::vector<std::span<const std::byte>> tx(net.size());
            std::vector<std::span<std::byte>> rx(net.size());
            for (unsigned p = 0; p < net.size(); ++p) {
                if (p != net.id()) {
                    tx[p] = context;
                    rx[p] = from[p];
                }
            }
            net.exchange_all(tx, rx);
            for (unsigned p = 0; p < net.size(); ++p) {
                if (p != net.id() && from[p] != context) {
                    throw std::runtime_error("parties have different preprocessing runs/plans");
                }
            }
        }

        unsigned parse_unsigned(const std::string &s) {
            unsigned x;
            const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), x);
            if (ec != std::errc{} || end != s.data() + s.size()) {
                throw std::invalid_argument("invalid unsigned option");
            }
            return x;
        }
    }

    int protocol_party_entry(int argc, char **argv) {
        try {
            std::string plan_path, material_path, input_path, hosts, open_override;
            unsigned id = UINT32_MAX, n = 0, local_repeats = 0, king_override = UINT32_MAX;
            for (int i = 1; i < argc; ++i) {
                std::string key = argv[i], value;
                const auto eq = key.find('=');
                if (eq != std::string::npos) {
                    value = key.substr(eq + 1);
                    key.resize(eq);
                } else {
                    if (i + 1 == argc) {
                        throw std::invalid_argument("party options require values");
                    }
                    value = argv[++i];
                }
                if (key == "--plan") {
                    plan_path = value;
                } else if (key == "--material") {
                    material_path = value;
                } else if (key == "--input") {
                    input_path = value;
                } else if (key == "--hosts") {
                    hosts = value;
                } else if (key == "--id") {
                    id = parse_unsigned(value);
                } else if (key == "--n") {
                    n = parse_unsigned(value);
                } else if (key == "--local-repeats") {
                    local_repeats = parse_unsigned(value);
                } else if (key == "--king") {
                    king_override = parse_unsigned(value);
                } else if (key == "--open-mode") {
                    parse_open_mode(value);
                    open_override = value;
                } else {
                    throw std::invalid_argument("unknown protocol party option");
                }
            }
            if (plan_path.empty() || material_path.empty() || hosts.empty()) {
                throw std::invalid_argument(
                    "party_main --id i --n n --hosts hosts.json --plan plan.json --material party_i.bin [--input private_i.json]");
            }
            Timer plan_timer;
            auto plan = MaterialPlan::load(plan_path);
            if (!plan.bitmuls.empty()) {
                throw std::invalid_argument(
                    "BitMul requires the workflow/API entry, not the standalone Boolean gate runner");
            }
            if (!plan.context.empty()) {
                throw std::invalid_argument(
                    "investigation plans require band_bench or compact_bench");
            }
            if (plan.n != n || id >= n) {
                throw std::invalid_argument("party plan/n/id mismatch");
            }
            auto config = Config::load(hosts);
            config.validate(n);
            config.king = king_override == UINT32_MAX ? plan.king : king_override;
            config.open_mode =
                parse_open_mode(open_override.empty() ? plan.open_mode : open_override);
            config.validate(n);
            omp_set_num_threads(static_cast<int>(config.threads));
            omp_set_dynamic(0);
            const auto plan_seconds = plan_timer.seconds();
            Timer load_timer;
            MaterialStore store(plan, id, material_path);
            const auto load_seconds = load_timer.seconds();
            Timer offline;
            store.claim_execution();
            Timer expand;
            store.preload();
            const auto expansion_seconds = expand.seconds();
            Timer input_prepare;
            InputManager inputs(store, plan, input_path);
            const auto input_prepare_seconds = input_prepare.seconds();
            Timer gate_prepare;
            std::vector<std::unique_ptr<PreparedGate>> gates;
            std::vector<Step *> steps;
            gates.reserve(plan.gates.size());
            steps.reserve(plan.gates.size());
            for (std::size_t g = 0; g < plan.gates.size(); ++g) {
                gates.push_back(prepare_gate(store, plan, g, inputs));
                steps.push_back(&gates.back()->step());
            }
            const auto gate_prepare_seconds = gate_prepare.seconds();
            RoundExecutor executor(steps);
            OpenLayout output_layout;
            const bool sum = plan.output_mode == "sum";
            for (const auto &g : plan.gates) {
                output_layout.add_ring(g.width, sum ? 1 : g.count);
            }
            output_layout.seal();
            AlignedBuffer<std::byte> output_share(output_layout.bytes()),
                output_value(output_layout.bytes());
            std::fill(output_share.span().begin(), output_share.span().end(), std::byte{0});
            std::fill(output_value.span().begin(), output_value.span().end(), std::byte{0});
            OpenBuf private_output(output_layout, output_share.span(), output_value.span());

            const auto offline_seconds = offline.seconds();
            Net net(id, config);
            check_session(net, store, plan);
            store.begin_online();
            net.reset_stats();
            Timer input_timer;
            inputs.run(net);
            const auto input_seconds = input_timer.seconds();
            const auto input_stats = network_stats(net);
            net.reset_stats();
            Timer online_timer;
            executor.run(net);
            const auto online_seconds = online_timer.seconds();
            const auto online_stats = network_stats(net);
            net.reset_stats();
            Timer output_timer;
            for (std::size_t g = 0; g < gates.size(); ++g) {
                const auto &f = output_layout.fields()[g];
                gates[g]->output_share(output_share.span().subspan(f.offset, f.bytes), sum);
            }
            net.reveal_to(plan.output_party, private_output);
            const auto output_seconds = output_timer.seconds();
            const auto output_stats = network_stats(net);
            const auto &layout = store.layout();
            Json result = {{"id", id},
                           {"n", n},
                           {"king", config.king},
                           {"open_mode", open_mode_name(config.open_mode)},
                           {"material_plan_open_mode", plan.open_mode},
                           {"material_plan_king", plan.king},
                           {"output_party", plan.output_party},
                           {"material_mode", material_mode_name(store.mode())},
                           {"plan_seconds", plan_seconds},
                           {"material_load_seconds", load_seconds},
                           {"offline_local_seconds", offline_seconds},
                           {"material_expansion_seconds", expansion_seconds},
                           {"input_prepare_seconds", input_prepare_seconds},
                           {"gate_precompute_seconds", gate_prepare_seconds},
                           {"input_seconds", input_seconds},
                           {"online_seconds", online_seconds},
                           {"output_seconds", output_seconds},
                           {"material_file_bytes", layout.file_bytes},
                           {"logical_share_bits", layout.logical_share_bits},
                           {"gate_preprocessing_bits", layout.gate_bits},
                           {"dense_share_bytes", layout.dense_share_bytes},
                           {"private_owner_mask_bytes", layout.owner_bytes},
                           {"prg_seed_bytes", layout.seed_bytes},
                           {"online_batch_bits", executor.plan().layout().bits()},
                           {"online_batch_bytes", executor.plan().layout().bytes()},
                           {"input", input_stats},
                           {"online", online_stats},
                           {"private_output", output_stats},
                           {"local_benchmarks", Json::object()}};
            if (id == plan.output_party) {
                Json values = Json::object();
                for (std::size_t g = 0; g < plan.gates.size(); ++g) {
                    const auto &f = output_layout.fields()[g];
                    auto bytes = output_value.span().subspan(f.offset, f.bytes);
                    Json out = Json::array();
                    for (std::size_t i = 0; i < f.count; ++i) {
                        if (f.width > 64) {
                            out.push_back(decimal_wide(load_wide(bytes, i * f.width, f.width)));
                        } else {
                            out.push_back(load_packed(bytes, i * f.width, f.width));
                        }
                    }
                    values[plan.gates[g].name] = out;
                }
                result["result"] = values;
            }

            for (std::size_t g = 0; g < gates.size(); ++g) {
                result["local_benchmarks"][plan.gates[g].name] =
                    gates[g]->benchmark(local_repeats, config.threads);
            }
            rusage usage{};
            if (::getrusage(RUSAGE_SELF, &usage)) {
                throw std::runtime_error("getrusage failed");
            }
            result["peak_rss_kib"] = usage.ru_maxrss;
            std::puts(result.dump().c_str());
            return 0;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "party_main: %s\n", e.what());
            return 1;
        }
    }
}
