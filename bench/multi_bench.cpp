#include "bj/ops/quicksort.hpp"
#include "../tests/allocation.hpp"
#include "bj/common/config.hpp"
#include "bj/common/timer.hpp"
#include "bj/plan/multi.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <sys/resource.h>
#include <time.h>

namespace {
    using namespace bj;
    using Json = nlohmann::json;

    Json net_json(const Net &net, double seconds) {
        Json peers = Json::array();
        for (unsigned i = 0; i < net.size(); ++i) {
            const auto &p = net.stats()[i];
            peers.push_back({{"peer", i},
                             {"sent_payload", p.sent_payload},
                             {"received_payload", p.received_payload},
                             {"sent_messages", p.sent_messages},
                             {"received_messages", p.received_messages}});
        }
        return {{"seconds", seconds},
                {"open_rounds", net.open_stats().rounds},
                {"hops", net.open_stats().hops},
                {"contribution_bits", net.open_stats().contribution_bits},
                {"peers", peers}};
    }

    Json stage_json(const StageStats &s) {
        std::uint64_t sent = 0, received = 0, wire_sent = 0, wire_received = 0;
        for (const auto &p : s.peers) {
            sent += p.sent_payload;
            received += p.received_payload;
            wire_sent += p.sent_wire;
            wire_received += p.received_wire;
        }
        return {{"seconds", s.seconds},
                {"poll_wait_seconds", s.io.poll_wait_seconds},
                {"socket_io_seconds", s.io.socket_io_seconds},
                {"caller_blocked_seconds", s.io.caller_blocked_seconds},
                {"local_compute_seconds",
                 std::max(0.0, s.seconds - s.io.poll_wait_seconds - s.io.socket_io_seconds)},
                {"open_rounds", s.open.rounds},
                {"open_hops", s.open.hops},
                {"contribution_bits", s.open.contribution_bits},
                {"sent_payload_bytes", sent},
                {"received_payload_bytes", received},
                {"sent_wire_bytes", wire_sent},
                {"received_wire_bytes", wire_received}};
    }
}

int main(int argc, char **argv) {
    try {
        std::string query_path, plan_path, material_path, input_path, hosts_path, compile_path,
            out_path, model_path, bank_path;
        unsigned id = UINT32_MAX, n = 0;
        double latency_ms = .1, bw_mbps = 1000;
        std::uint64_t budget = UINT64_MAX;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("multi options require values");
            }
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "--query") {
                query_path = v;
            } else if (k == "--plan") {
                plan_path = v;
            } else if (k == "--material") {
                material_path = v;
            } else if (k == "--bank-file") {
                bank_path = v;
            } else if (k == "--input") {
                input_path = v;
            } else if (k == "--hosts") {
                hosts_path = v;
            } else if (k == "--id") {
                id = static_cast<unsigned>(std::stoul(v));
            } else if (k == "--n") {
                n = static_cast<unsigned>(std::stoul(v));
            } else if (k == "--compile") {
                compile_path = v;
            } else if (k == "--out") {
                out_path = v;
            } else if (k == "--model") {
                model_path = v;
            } else if (k == "--latency-ms") {
                latency_ms = std::stod(v);
            } else if (k == "--bandwidth-mbps") {
                bw_mbps = std::stod(v);
            } else if (k == "--budget-bits") {
                budget = std::stoull(v);
            } else {
                throw std::invalid_argument("unknown multi option: " + k);
            }
        }
        if (!model_path.empty()) {
            const auto model = enumerate_operators(
                Plan::load(model_path), {latency_ms / 1000, bw_mbps * 1e6 / 8}, budget);
            std::puts(model.dump().c_str());
            return 0;
        }
        if (!compile_path.empty()) {
            const auto cfg = Plan::load(compile_path);
            const auto c = compile_multi(cfg);
            if (out_path.empty()) {
                throw std::invalid_argument("--compile requires --out material-plan.json");
            }
            std::ofstream out(out_path);
            if (!out) {
                throw std::runtime_error("cannot write material plan");
            }
            out << c.material.canonical() << '\n';
            MaterialLayout layout(c.material, 0, MaterialMode::Prg);
            Json inputs = Json::array();
            for (const auto &x : c.material.inputs) {
                inputs.push_back(
                    {{"name", x.name}, {"owner", x.owner}, {"width", x.width}, {"count", x.count}});
            }
            std::puts(Json{{"shared_threshold_tables", c.shared_threshold_tables},
                           {"time_groups", c.time_groups},
                           {"group_members", c.group_members},
                           {"logical_share_bits", layout.logical_share_bits},
                           {"inputs", inputs},
                           {"canonical_plan", cfg.json()}}
                          .dump()
                          .c_str());
            return 0;
        }
        if (query_path.empty() || plan_path.empty() || material_path.empty() ||
            hosts_path.empty()) {
            throw std::invalid_argument(
                "multi_bench --query query.json --plan material-plan.json --material "
                "party_i.bin --input private_i.json --hosts hosts.json --id i --n n");
        }
        const auto cfg = Plan::load(query_path);
        if (n != cfg.common.n || id >= n) {
            throw std::invalid_argument("multi party/n mismatch");
        }
        const auto plan = MaterialPlan::load(plan_path);
        auto conf = Config::load(hosts_path);
        conf.king = cfg.common.king;
        conf.open_mode = parse_open_mode(cfg.common.open_mode);
        conf.validate(n);
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(conf.threads));
        Timer timer;
        MaterialStore store(plan, id, material_path);
        store.claim_execution();
        store.preload();
        InputManager input(store, plan, input_path, bank_path);
        MultiQuery multi(cfg, store, plan, input);
        OpenLayout output_layout;
        output_layout.add_ring(64, 1);
        for (const auto &q : cfg.templates) {
            if (q.config.output == "count") {
                output_layout.add_ring(64, 1);
            } else {
                output_layout.add_bits(1);
            }
        }
        output_layout.seal();
        AlignedBuffer<std::byte> txout(output_layout.bytes()), rxout(output_layout.bytes());
        std::fill(txout.span().begin(), txout.span().end(), std::byte{0});
        std::fill(rxout.span().begin(), rxout.span().end(), std::byte{0});
        OpenBuf output(output_layout, txout.span(), rxout.span());
        Net net(id, conf);
        std::array<std::byte, 24> context{};
        std::memcpy(context.data(), store.run_id().data(), 16);
        const auto fingerprint = plan.fingerprint();
        std::memcpy(context.data() + 16, &fingerprint, 8);
        std::vector<std::array<std::byte, 24>> received(n);
        std::vector<std::span<const std::byte>> send(n);
        std::vector<std::span<std::byte>> recv(n);
        for (unsigned peer = 0; peer < n; ++peer) {
            if (peer != id) {
                send[peer] = context;
                recv[peer] = received[peer];
            }
        }
        net.exchange_all(send, recv);
        for (unsigned peer = 0; peer < n; ++peer) {
            if (peer != id && received[peer] != context) {
                throw std::runtime_error("different multi preprocessing runs/plans");
            }
        }
        multi.prepare(net, conf.threads);
        const auto offline = timer.seconds();
        StageStats input_stage(n), online_stage(n), output_stage(n);
        store.begin_online();
        net.reset_stats();
        timer.reset();
        test::allocations = 0;
        test::count_allocations = true;
        Timer total_online_timer;
        input.run(net);
        const auto input_seconds = timer.seconds();
        const auto input_rounds = net.open_stats();
        input_stage.capture(net, input_seconds);
        net.reset_stats();
        timer.reset();
        multi.run(net);
        const auto online_seconds = timer.seconds();
        const auto online_stats = net.open_stats();
        online_stage.capture(net, online_seconds);
        Timer output_timer;
        const auto omega = multi.overflow();
        pack_native<std::uint64_t>(
            64, std::span<const std::uint64_t>(&omega, 1), txout.span().first(8));
        for (std::size_t i = 0; i < cfg.templates.size(); ++i) {
            const auto &f = output_layout.fields()[i + 1];
            if (f.width) {
                pack_native<std::uint64_t>(
                    64, multi.counts(i), txout.span().subspan(f.offset, f.bytes));
            } else {
                store_bits(txout.span().subspan(f.offset), 0, 1, multi.alarms(i));
            }
        }
        net.reveal_to(cfg.common.initiator, output);
        timespec receipt{};
        ::clock_gettime(CLOCK_MONOTONIC, &receipt);
        const std::uint64_t ready_ns = std::uint64_t(receipt.tv_sec) * 1000000000 + receipt.tv_nsec;
        const auto online_total_seconds = total_online_timer.seconds();
        const auto output_seconds = output_timer.seconds();
        output_stage.capture(net, output_seconds, &online_stage);
        test::count_allocations = false;
        const auto allocations = test::allocations.load();
        const auto stats = multi.stats();
        rusage usage{};
        ::getrusage(RUSAGE_SELF, &usage);
        Json result = {{"online_total_seconds", online_total_seconds},
                       {"output_ready_monotonic_ns", ready_ns},
                       {"id", id},
                       {"n", n},
                       {"king", conf.king},
                       {"open_mode", open_mode_name(conf.open_mode)},
                       {"offline_seconds", offline},
                       {"input_seconds", input_seconds},
                       {"online_seconds", online_seconds},
                       {"private_output_seconds", output_seconds},
                       {"timed_total_seconds", input_seconds + online_seconds + output_seconds},
                       {"input_open_rounds", input_rounds.rounds},
                       {"online_new_calls", allocations},
                       {"open_rounds", online_stats.rounds},
                       {"open_hops", online_stats.hops},
                       {"contribution_bits", online_stats.contribution_bits},
                       {"network", net_json(net, online_seconds)},
                       {"logical_share_bits", store.layout().logical_share_bits},
                       {"file_bytes", store.layout().file_bytes},
                       {"peak_rss_kib", usage.ru_maxrss},
                       {"compact",
                        {{"open_rounds", multi.compact_stats().open_rounds},
                         {"logical_rounds", multi.compact_stats().logical_rounds},
                         {"shuffle_bytes", multi.compact_stats().shuffle_bytes}}},
                       {"schedule",
                        {{"unmerged_opens", stats.unmerged_opens},
                         {"merged_opens", stats.merged_opens},
                         {"unmerged_shuffle_hops", stats.unmerged_shuffle_hops},
                         {"merged_shuffle_hops", stats.merged_shuffle_hops},
                         {"graph_rounds", stats.graph_rounds},
                         {"contribution_bits", stats.contribution_bits},
                         {"shuffle_bytes", stats.shuffle_bytes},
                         {"wave_rounds", stats.wave_rounds}}}};
        result["stages"] = {{"input", stage_json(input_stage)},
                            {"filter_union", stage_json(multi.union_stage())},
                            {"compact", stage_json(multi.compact_stage())},
                            {"operators", stage_json(multi.operator_stage())},
                            {"private_output", stage_json(output_stage)}};
        result["templates"] = Json::array();
        for (std::size_t i = 0; i < cfg.templates.size(); ++i) {
            result["templates"].push_back(
                {{"name", cfg.templates[i].name},
                 {"pattern", cfg.templates[i].config.pattern},
                 {"operator", cfg.templates[i].config.algorithm},
                 {"seconds_including_dependency_wait", multi.template_seconds(i)}});
        }
        if (id == cfg.common.initiator) {
            Json values = Json::object();
            for (std::size_t i = 0; i < cfg.templates.size(); ++i) {
                const auto &f = output_layout.fields()[i + 1];
                values[cfg.templates[i].name] =
                    load_packed(rxout.span().subspan(f.offset), 0, f.width ? 64 : 1);
            }
            result["result"] = {{"omega", load_packed(rxout.span(), 0, 64)}, {"templates", values}};
        }
        std::puts(result.dump().c_str());
        return 0;
    } catch (const bj::MaterialBudgetExhausted &e) {
        bj::test::count_allocations = false;
        nlohmann::json failure = {{"aborted", 1},
                                  {"status", "material_abort"},
                                  {"output", "bottom"},
                                  {"reason", e.what()}};
        std::puts(failure.dump().c_str());
        return 0;
    } catch (const std::exception &e) {
        bj::test::count_allocations = false;
        std::fprintf(stderr, "multi_bench: %s\n", e.what());
        return 1;
    }
}
