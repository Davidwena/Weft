#include "bj/ops/batch_suspects.hpp"
#include "bj/ops/nl.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <set>
#include <omp.h>
#include <sys/resource.h>
#include <time.h>
using namespace bj;
using Json = nlohmann::json;

Json read(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    return Json::parse(f);
}

Json sj(const StageStats &s) {
    std::uint64_t tx = 0, rx = 0, wire = 0;
    for (auto &p : s.peers) {
        tx += p.sent_payload;
        rx += p.received_payload;
        wire += p.sent_wire;
    }
    return {{"seconds", s.seconds},
            {"open_rounds", s.open.rounds},
            {"hops", s.open.hops},
            {"contribution_bits", s.open.contribution_bits},
            {"sent_payload_bytes", tx},
            {"sent_wire_bytes", wire},
            {"received_payload_bytes", rx}};
}

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) {
                throw std::invalid_argument("key/value options");
            }
            args[argv[i]] = argv[i + 1];
        }
        auto doc = read(args.at("--query"));
        auto c = InvestigationConfig::parse(doc);
        const std::size_t S = doc.value("num_suspects", std::size_t{1});
        const bool legacy = args.contains("--legacy");
        if (legacy && S != 1) {
            throw std::invalid_argument("legacy requires S=1");
        }
        auto p = legacy ? compile_investigation(c) : compile_batch_suspects(c, S);
        if (args.contains("--compile")) {
            std::ofstream(args.at("--compile")) << p.canonical() << '\n';
            return 0;
        }
        if (args.contains("--estimate")) {
            auto mode = parse_material_mode(args.at("--estimate"));
            Json sizes = Json::array(), logical = Json::array(), dense = Json::array();
            for (unsigned i = 0; i < c.n; ++i) {
                MaterialLayout layout(p, i, mode);
                sizes.push_back(layout.file_bytes);
                logical.push_back(layout.logical_share_bits);
                dense.push_back(layout.dense_share_bytes);
            }
            std::cout << Json{{"file_bytes_per_bank", sizes},
                              {"logical_share_bits_per_bank", logical},
                              {"dense_share_bytes_per_bank", dense},
                              {"num_suspects", S},
                              {"table_rows", c.size() + S * (c.kin + c.kout)}}
                             .dump()
                      << '\n';
            return 0;
        }
        const auto id = static_cast<unsigned>(std::stoul(args.at("--id")));
        auto hosts = Config::load(args.at("--hosts"));
        hosts.king = c.king;
        hosts.open_mode = parse_open_mode(c.open_mode);
        hosts.validate(c.n);
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(hosts.threads));

        if (id == c.initiator) {
            auto private_values = read(args.at("--input")).at("inputs");
            std::set<std::uint64_t> seen;
            for (std::size_t j = 0; j < S; ++j) {
                auto name = legacy ? "subject" : "suspect" + std::to_string(j) + ".subject";
                auto v = private_values.at(name);
                if (!v.is_array() || v.size() != 1 || !v[0].is_number_unsigned()) {
                    throw std::invalid_argument("invalid private suspect input");
                }
                if (!seen.insert(v[0].get<std::uint64_t>()).second) {
                    throw std::invalid_argument(
                        "duplicate suspects: local rejection before network");
                }
            }
        }
        if (args.contains("--validate-private-only")) {
            std::cout << "{\"valid\":true}" << '\n';
            return 0;
        }
        Timer prep;
        MaterialStore store(p, id, args.at("--material"));
        store.claim_execution();
        store.preload();
        const auto load = prep.seconds();
        InputManager input(store, p, args.at("--input"), args.at("--bank-file"));
        std::unique_ptr<Compact> compact;
        std::unique_ptr<NestedLoop> nl;
        std::unique_ptr<BatchSuspects> batch;
        if (legacy) {
            compact = std::make_unique<Compact>(c, store, p, input);
            nl = std::make_unique<NestedLoop>(c,
                                              store,
                                              p,
                                              compact->inbound(),
                                              compact->outbound(),
                                              compact->omega_in(),
                                              compact->omega_out());
        } else {
            batch = std::make_unique<BatchSuspects>(c, S, store, p, input);
        }
        Net net(id, hosts);
        std::array<std::byte, 24> context{};
        std::memcpy(context.data(), store.run_id().data(), 16);
        auto fp = p.fingerprint();
        std::memcpy(context.data() + 16, &fp, 8);
        std::vector<std::array<std::byte, 24>> received(c.n);
        std::vector<std::span<const std::byte>> tx(c.n);
        std::vector<std::span<std::byte>> rx(c.n);
        for (unsigned b = 0; b < c.n; ++b) {
            if (b != id) {
                tx[b] = context;
                rx[b] = received[b];
            }
        }
        net.exchange_all(tx, rx);
        for (unsigned b = 0; b < c.n; ++b) {
            if (b != id && received[b] != context) {
                throw std::runtime_error("different batch plans or preprocessing runs");
            }
        }
        if (batch) {
            batch->prepare(net, hosts.threads);
        }
        OpenLayout ol;
        ol.add_ring(64, 2 * S);
        ol.seal();
        AlignedBuffer<std::byte> out(ol.bytes()), answer(ol.bytes());
        OpenBuf ob(ol, out.span(), answer.span());
        AlignedBuffer<std::uint64_t> values(2 * S);
#pragma omp parallel
        { asm volatile("" ::: "memory"); }
        auto preparation = prep.seconds();
        store.begin_online();
        net.reset_stats();
        Timer online, timer;
        input.run(net);
        StageStats inputstats(c.n);
        inputstats.capture(net, timer.seconds());
        net.reset_stats();
        timer.reset();
        StageStats compactstats(c.n), joinstats(c.n);
        std::vector<std::uint64_t> counts, omegas;
        Json stages = Json::object();
        Json slotcounts;
        if (legacy) {
            compact->run(net);
            compactstats.capture(net, timer.seconds());
            net.reset_stats();
            timer.reset();
            nl->run(net);
            joinstats.capture(net, timer.seconds());
            counts = {nl->count_shares()[0]};
            omegas = {nl->overflow_share()};
            auto cs = compact->stats();
            stages["compact_eq"] = {{"seconds", cs.seconds[0] + cs.seconds[1]}, {"open_rounds", 2}};
            stages["compact_prefix_msb"] = {{"seconds", cs.seconds[2] + cs.seconds[3]},
                                            {"open_rounds", 6}};
            stages["compact_shuffle_open"] = {
                {"seconds", cs.seconds[4] + cs.seconds[5] + cs.seconds[6]}, {"open_rounds", 2}};
            stages["compact_remask"] = {{"seconds", cs.seconds[7]}, {"open_rounds", 1}};
            slotcounts = Json::array({{c.kin, c.kout}});
        } else {
            batch->run(net);
            joinstats.capture(net, 0);
            auto bs = batch->stages();
            const std::array<std::string, 4> names{
                "compact_eq", "compact_prefix_msb", "compact_shuffle_open", "compact_remask"};
            double comp = 0;
            for (unsigned i = 0; i < 4; ++i) {
                stages[names[i]] = sj(bs[i]);
                comp += bs[i].seconds;
            }
            joinstats.seconds = timer.seconds() - comp;
            counts = batch->count_shares();
            omegas = batch->overflow_shares();
            slotcounts = batch->slot_counts();
        }
        stages["input"] = sj(inputstats);
        stages["join"] = sj(joinstats);
        net.reset_stats();
        Timer output_timer;
        for (std::size_t j = 0; j < S; ++j) {
            values.data()[2 * j] = counts.at(j);
            values.data()[2 * j + 1] = omegas.at(j);
        }
        pack_native<std::uint64_t>(64, values.span(), out.span());
        net.reveal_to(c.initiator, ob);
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const auto ready = std::uint64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
        auto on = online.seconds();
        StageStats outputstats(c.n);
        outputstats.capture(net, output_timer.seconds());
        stages["output"] = sj(outputstats);
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        Json result = {{"id", id},
                       {"S", S},
                       {"legacy", legacy},
                       {"material_load_seconds", load},
                       {"preparation_seconds", preparation},
                       {"online_seconds", on},
                       {"output_ready_monotonic_ns", ready},
                       {"stages", stages},
                       {"slot_counts", slotcounts},
                       {"peak_rss_bytes", std::uint64_t(usage.ru_maxrss) * 1024},
                       {"file_bytes", store.layout().file_bytes},
                       {"logical_share_bits", store.layout().logical_share_bits},
                       {"environment",
                        {{"OMP_WAIT_POLICY",
                          std::getenv("OMP_WAIT_POLICY") ? std::getenv("OMP_WAIT_POLICY") : ""},
                         {"GOMP_SPINCOUNT",
                          std::getenv("GOMP_SPINCOUNT") ? std::getenv("GOMP_SPINCOUNT") : ""}}}};
        if (legacy) {
            result["compact_total"] = sj(compactstats);
        }
        if (id == c.initiator) {
            unpack_native<std::uint64_t>(64, answer.span(), values.span());
            result["results"] = Json::array();
            for (std::size_t j = 0; j < S; ++j) {
                result["results"].push_back(
                    {{"aggregate", values.data()[2 * j]}, {"omega", values.data()[2 * j + 1]}});
            }
        }
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "batch_suspects: " << e.what() << '\n';
        return 1;
    }
}
