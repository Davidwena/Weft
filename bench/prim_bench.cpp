#include <time.h>
#include "bj/core/input.hpp"
#include "bj/proto/msb.hpp"
#include "bj/proto/eda.hpp"
#include "bj/proto/eq.hpp"
#include "bj/proto/lts.hpp"
#include "bj/common/timer.hpp"
#include "bj/common/config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>
#include <charconv>
#include <cstring>
#include <omp.h>
#include <sys/resource.h>

namespace {
    using namespace bj;
    using Json = nlohmann::json;

    struct Runtime {
        BitVec output;

        explicit Runtime(std::size_t n) : output(n) {
        }

        virtual ~Runtime() = default;

        virtual std::size_t stage_count() {
            return opening() ? 2 : 1;
        }

        virtual Step &stage(std::size_t i) {
            return opening() && i == 0 ? *opening() : comparison();
        }

        virtual Step *opening() = 0;
        virtual Step &comparison() = 0;
        virtual void evaluate() = 0;
    };

    template <class T> struct NativeRuntime : Runtime {
        std::unique_ptr<EdaBatch<T>> eda;
        std::unique_ptr<LTBitsBatch<T>> lt;
        std::unique_ptr<MsbKnown<T>> msbk;
        std::unique_ptr<MsbFresh<T>> msbf;
        std::unique_ptr<EqComparison<T>> eq;
        std::unique_ptr<EqFresh<T>> eqf;
        std::unique_ptr<LtsBatch<T>> lts;

        static std::span<const T> values(const PreparedInput &i, bool shares) {
            if constexpr (sizeof(T) == 4) {
                return shares ? i.share32.span() : i.delta32.span();
            } else if constexpr (sizeof(T) == 8) {
                return shares ? i.share64.span() : i.delta64.span();
            } else {
                return shares ? i.share128.span() : i.delta128.span();
            }
        }

        NativeRuntime(MaterialStore &s,
                      const MaterialPlan &p,
                      std::size_t c,
                      unsigned e,
                      const InputManager &in,
                      bool reverse,
                      std::shared_ptr<const BlockTable> table)
            : Runtime(p.comparisons[c].count) {
            const auto &spec = p.comparisons[c];
            const bool fresh = spec.kind == CompareKind::EqFresh ||
                               spec.kind == CompareKind::MsbFresh ||
                               spec.kind == CompareKind::LtsFresh ||
                               spec.kind == CompareKind::EdaMsb || spec.kind == CompareKind::EdaEq;
            const auto x = values(in.at(spec.inputs[0]), fresh);
            switch (spec.kind) {
            case CompareKind::EdaMsb:
            case CompareKind::EdaEq:
                eda = std::make_unique<EdaBatch<T>>(s, p, c, x, output.words());
                break;
            case CompareKind::LTBits:
                lt =
                    std::make_unique<LTBitsBatch<T>>(s, p, c, e, x, output.words(), reverse, table);
                break;
            case CompareKind::MsbKnown:
                msbk = std::make_unique<MsbKnown<T>>(s, p, c, e, x, output.words(), table);
                break;
            case CompareKind::MsbFresh:
                msbf = std::make_unique<MsbFresh<T>>(s, p, c, x, output.words());
                break;
            case CompareKind::EqFresh:
                eqf = std::make_unique<EqFresh<T>>(s, p, c, x, output.words());
                break;
            case CompareKind::Eq:
                eq = std::make_unique<EqComparison<T>>(s, p, c, e, x, output.words(), table);
                break;
            default:
                lts = std::make_unique<LtsBatch<T>>(
                    s, p, c, x, values(in.at(spec.inputs[1]), fresh), output.words());
                break;
            }
        }

        std::size_t stage_count() override {
            return eda ? eda->stage_count() : Runtime::stage_count();
        }

        Step &stage(std::size_t i) override {
            return eda ? eda->stage(i) : Runtime::stage(i);
        }

        Step *opening() override {
            if (eqf) {
                return &eqf->opening();
            }
            if (msbf) {
                return &msbf->opening();
            }
            if (lts) {
                return lts->opening();
            }
            return nullptr;
        }

        Step &comparison() override {
            if (eqf) {
                return eqf->comparison();
            }
            if (lt) {
                return *lt;
            }
            if (msbk) {
                return *msbk;
            }
            if (msbf) {
                return msbf->comparison();
            }
            if (eq) {
                return *eq;
            }
            return lts->comparison();
        }

        void evaluate() override {
            if (eda || eqf) {
                return;
            }
            if (lt) {
                lt->evaluate_local(output.words());
            } else if (msbk) {
                msbk->evaluate_local(output.words());
            } else if (msbf) {
                msbf->evaluate_local(output.words());
            } else if (eq) {
                eq->evaluate_local(output.words());
            } else {
                lts->evaluate_local(output.words());
            }
        }
    };

    Json stats(const Net &net, double seconds) {
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
        const auto io = net.io_timing();
        return {{"peers", peers},
                {"open_rounds", net.open_stats().rounds},
                {"network_hops", net.open_stats().hops},
                {"contribution_bits", net.open_stats().contribution_bits},
                {"seconds", seconds},
                {"poll_wait_seconds", io.poll_wait_seconds},
                {"socket_io_seconds", io.socket_io_seconds},
                {"caller_blocked_seconds", io.caller_blocked_seconds},
                {"local_compute_seconds",
                 std::max(0.0, seconds - io.poll_wait_seconds - io.socket_io_seconds)}};
    }

    unsigned number(const std::string &s) {
        unsigned n = 0;
        auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), n);
        if (ec != std::errc{} || end != s.data() + s.size()) {
            throw std::invalid_argument("numeric option");
        }
        return n;
    }
}

int main(int argc, char **argv) {
    try {
        std::string plan_file, material_file, input_file, hosts, result_file, mode_override;
        unsigned id = UINT32_MAX, n = 0, repeats = 0, king = UINT32_MAX;
        bool reverse = false;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i], v;
            auto eq = k.find('=');
            if (eq != std::string::npos) {
                v = k.substr(eq + 1);
                k.resize(eq);
            } else {
                if (i + 1 == argc) {
                    throw std::invalid_argument("option requires value");
                }
                v = argv[++i];
            }
            if (k == "--plan") {
                plan_file = v;
            } else if (k == "--material") {
                material_file = v;
            } else if (k == "--input") {
                input_file = v;
            } else if (k == "--hosts") {
                hosts = v;
            } else if (k == "--result-file") {
                result_file = v;
            } else if (k == "--id") {
                id = number(v);
            } else if (k == "--n") {
                n = number(v);
            } else if (k == "--local-repeats") {
                repeats = number(v);
            } else if (k == "--reverse") {
                reverse = number(v) != 0;
            } else if (k == "--king") {
                king = number(v);
            } else if (k == "--open-mode") {
                mode_override = v;
            } else {
                throw std::invalid_argument("unknown prim_bench option");
            }
        }
        Timer plan_time;
        const auto plan = MaterialPlan::load(plan_file);
        if (plan.n != n || id >= n || plan.comparisons.empty() || !plan.gates.empty()) {
            throw std::invalid_argument(
                "prim_bench requires comparison-only plan and correct party id/n");
        }
        auto config = Config::load(hosts);
        config.king = king == UINT32_MAX ? plan.king : king;
        config.open_mode = parse_open_mode(mode_override.empty() ? plan.open_mode : mode_override);
        config.validate(n);
        omp_set_num_threads(static_cast<int>(config.threads));
        omp_set_dynamic(0);
        const auto plan_seconds = plan_time.seconds();
        Timer loading;
        MaterialStore store(plan, id, material_file);
        const auto load_seconds = loading.seconds();
        Timer offline;
        store.claim_execution();
        store.preload();
        InputManager inputs(store, plan, input_file);
        std::vector<std::unique_ptr<Runtime>> batches;
        std::vector<std::vector<Step *>> layers;
        std::vector<std::size_t> offsets;
        std::size_t output_bits = 0;
        for (std::size_t c = 0; c < plan.comparisons.size(); ++c) {
            const auto &spec = plan.comparisons[c];
            std::shared_ptr<const BlockTable> table;
            if (spec.kind == CompareKind::LTBits || spec.kind == CompareKind::Eq ||
                (spec.kind == CompareKind::MsbKnown && spec.width > 1)) {
                table = load_block_table(store, plan, c);
            }
            for (unsigned e = 0; e < spec.evaluations; ++e) {
                if (spec.width <= 32) {
                    batches.push_back(std::make_unique<NativeRuntime<std::uint32_t>>(
                        store, plan, c, e, inputs, reverse, table));
                } else if (spec.width <= 64) {
                    batches.push_back(std::make_unique<NativeRuntime<std::uint64_t>>(
                        store, plan, c, e, inputs, reverse, table));
                } else {
                    batches.push_back(std::make_unique<NativeRuntime<uint128_t>>(
                        store, plan, c, e, inputs, reverse, table));
                }
                auto &batch = *batches.back();
                if (layers.size() < batch.stage_count()) {
                    layers.resize(batch.stage_count());
                }
                for (std::size_t l = 0; l < batch.stage_count(); ++l) {
                    layers[l].push_back(&batch.stage(l));
                }
                offsets.push_back(output_bits);
                output_bits += spec.count;
            }
        }
        std::vector<std::unique_ptr<RoundExecutor>> rounds;
        for (auto &layer : layers) {
            rounds.push_back(std::make_unique<RoundExecutor>(layer));
        }
        OpenLayout out_layout;
        out_layout.add_bits(output_bits);
        out_layout.seal();
        AlignedBuffer<std::byte> out_share(out_layout.bytes()), out_value(out_layout.bytes());
        std::fill(out_share.span().begin(), out_share.span().end(), std::byte{0});
        std::fill(out_value.span().begin(), out_value.span().end(), std::byte{0});
        OpenBuf out(out_layout, out_share.span(), out_value.span());
        const auto offline_seconds = offline.seconds();
        Net net(id, config);
        std::array<std::byte, 24> context{};
        std::memcpy(context.data(), store.run_id().data(), 16);
        const auto hash = plan.fingerprint();
        std::memcpy(context.data() + 16, &hash, 8);
        std::vector<std::array<std::byte, 24>> received(n);
        std::vector<std::span<const std::byte>> tx(n);
        std::vector<std::span<std::byte>> rx(n);
        for (unsigned p = 0; p < n; ++p) {
            if (p != id) {
                tx[p] = context;
                rx[p] = received[p];
            }
        }
        net.exchange_all(tx, rx);
        for (unsigned p = 0; p < n; ++p) {
            if (p != id && received[p] != context) {
                throw std::runtime_error("mixed material run");
            }
        }
        store.begin_online();
        net.reset_stats();
        Timer input_time;
        inputs.run(net);
        const auto input_seconds = input_time.seconds();
        const auto input_stats = stats(net, input_seconds);
        net.reset_stats();
        Timer online;
        for (auto &round : rounds) {
            if (round->plan().layout().bits() == 0) {
                round->run_without_open();
            } else {
                round->run(net);
            }
        }
        const auto online_seconds = online.seconds();
        const auto online_stats = stats(net, online_seconds);
        net.reset_stats();
        Timer output_time;
        std::size_t index = 0;
        for (const auto &spec : plan.comparisons) {
            for (unsigned e = 0; e < spec.evaluations; ++e) {
                store_bits(
                    out_share.span(), offsets[index], spec.count, batches[index]->output.words());
                ++index;
            }
        }
        net.reveal_to(plan.output_party, out);
        timespec receipt{};
        clock_gettime(CLOCK_MONOTONIC, &receipt);
        const std::uint64_t ready_ns = std::uint64_t(receipt.tv_sec) * 1000000000 + receipt.tv_nsec;
        const auto output_seconds = output_time.seconds();
        const auto output_stats = stats(net, output_seconds);
        Json report = {{"output_ready_monotonic_ns", ready_ns},
                       {"id", id},
                       {"n", n},
                       {"king", config.king},
                       {"open_mode", open_mode_name(config.open_mode)},
                       {"reverse", reverse},
                       {"material_mode", material_mode_name(store.mode())},
                       {"plan_seconds", plan_seconds},
                       {"material_load_seconds", load_seconds},
                       {"offline_local_seconds", offline_seconds},
                       {"input_seconds", input_seconds},
                       {"online_seconds", online_seconds},
                       {"output_seconds", output_seconds},
                       {"gate_preprocessing_bits", store.layout().gate_bits},
                       {"logical_share_bits", store.layout().logical_share_bits},
                       {"material_file_bytes", store.layout().file_bytes},
                       {"dense_share_bytes", store.layout().dense_share_bytes},
                       {"prg_seed_bytes", store.layout().seed_bytes},
                       {"input", input_stats},
                       {"online", online_stats},
                       {"private_output", output_stats},
                       {"local_benchmarks", Json::array()}};
        if (id == plan.output_party) {
            if (!result_file.empty()) {
                std::ofstream file(result_file, std::ios::binary);
                file.write(reinterpret_cast<const char *>(out_value.data()),
                           static_cast<std::streamsize>(out_value.size()));
                if (!file) {
                    throw std::runtime_error("cannot write private result");
                }
            }
            if (output_bits <= 4096) {
                Json bits = Json::array();
                for (std::size_t i = 0; i < output_bits; ++i) {
                    bits.push_back(load_packed(out_value.span(), i, 1));
                }
                report["result"] = bits;
            }
        }
        bool has_baseline = false;
        for (const auto &c : plan.comparisons) {
            has_baseline |= c.kind == CompareKind::EdaMsb || c.kind == CompareKind::EdaEq;
        }
        if (repeats && !has_baseline) {
            for (unsigned threads : {1u, config.threads}) {
                omp_set_num_threads(static_cast<int>(threads));
                for (auto &batch : batches) {
                    batch->evaluate();
                }
                Timer local;
                for (unsigned r = 0; r < repeats; ++r) {
                    for (auto &batch : batches) {
                        batch->evaluate();
                    }
                }
                const auto seconds = local.seconds();
                std::uint64_t sink = 0;
                for (const auto &batch : batches) {
                    for (auto word : batch->output.words()) {
                        sink ^= word;
                    }
                }
                asm volatile("" : : "r"(sink) : "memory");
                report["local_benchmarks"].push_back(
                    {{"threads", threads},
                     {"repeats", repeats},
                     {"seconds", seconds},
                     {"instances_per_second",
                      seconds ? static_cast<double>(output_bits) * repeats / seconds : 0}});
            }
        }
        rusage usage{};
        ::getrusage(RUSAGE_SELF, &usage);
        report["peak_rss_kib"] = usage.ru_maxrss;
        std::puts(report.dump().c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "prim_bench: %s\n", e.what());
        return 1;
    }
}
