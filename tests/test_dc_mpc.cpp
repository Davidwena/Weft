#include "bj/ops/compact.hpp"
#include "bj/ops/dc.hpp"
#include "bj/ops/nl.hpp"
#include "bj/ref/band.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
    using namespace bj;

    std::vector<Host> dc_hosts(unsigned n) {
        std::vector<Host> result;
        std::vector<int> sockets;
        for (unsigned i = 0; i < n; ++i) {
            const auto fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                throw std::runtime_error("socket");
            }
            sockets.push_back(fd);
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a))) {
                throw std::runtime_error("bind");
            }
            socklen_t length = sizeof(a);
            ::getsockname(fd, reinterpret_cast<sockaddr *>(&a), &length);
            result.push_back({"127.0.0.1", ntohs(a.sin_port)});
        }
        for (auto fd : sockets) {
            ::close(fd);
        }
        return result;
    }

    TEST(DCMPC, NLAndDCConsumeIdenticalCompressedSlotObjects) {
        omp_set_num_threads(2);
        for (std::size_t K : {16u, 32u, 64u, 128u, 256u, 512u}) {
            char name[] = "/tmp/dc-same-slots-XXXXXX";
            const auto tmp = ::mkdtemp(name);
            ASSERT_NE(tmp, nullptr);
            const std::filesystem::path directory(tmp);

            struct Cleanup {
                std::filesystem::path path;

                ~Cleanup() {
                    std::error_code e;
                    std::filesystem::remove_all(path, e);
                }
            } cleanup{directory};

            InvestigationConfig c;
            c.n = 3;
            c.initiator = 2;
            c.rows = {K, K, K};
            c.kin = c.kout = K;
            c.algorithm = "dc";
            c.p = 3;
            c.q = 4;
            c.bq = 3;
            c.ba = 8;
            c.ka = 13;
            c.windows = {{0, 5}};
            auto plan = compile_investigation(c);
            auto nlconfig = c;
            nlconfig.algorithm = "nl";
            auto nlplan = compile_investigation(nlconfig);

            ASSERT_EQ(plan.inputs.size(), nlplan.inputs.size());
            const auto shift = plan.masks.size();
            auto remap = [&](std::size_t id) {
                return id < plan.inputs.size() ? id : id + shift;
            };
            for (auto m : nlplan.masks) {
                for (auto &t : m.terms) {
                    t.source = remap(t.source);
                }
                if (m.name.starts_with("pair.")) {
                    plan.masks.push_back(std::move(m));
                }
            }

            for (auto &m : plan.masks) {
                if (m.name.starts_with("pair.")) {
                    for (auto &t : m.terms) {
                        const auto original = t.source - shift;
                        const auto &source =
                            original < nlplan.inputs.size()
                                ? nlplan.inputs[original].name
                                : nlplan.masks[original - nlplan.inputs.size()].name;
                        t.source = named_mask(plan, source);
                    }
                }
            }
            for (auto cmp : nlplan.comparisons) {
                if (cmp.name.starts_with("band.")) {
                    for (auto &id : cmp.inputs) {
                        const auto &m = nlplan.masks[id - nlplan.inputs.size()];
                        id = named_mask(plan, m.name);
                    }
                    plan.comparisons.push_back(std::move(cmp));
                }
            }
            for (auto gate : nlplan.gates) {
                if (gate.name.starts_with("band.") || gate.name == "output.omega") {
                    if (gate.name == "output.omega") {
                        gate.name = "output.omega.nl";
                    }
                    plan.gates.push_back(std::move(gate));
                }
            }
            plan.validate();
            generate_materials(plan, MaterialMode::Prg, directory / "m");
            std::vector<ref::PlainRow> plain;
            std::array<std::unique_ptr<MaterialStore>, 3> stores;
            std::array<std::unique_ptr<InputManager>, 3> inputs;
            std::array<std::unique_ptr<Compact>, 3> compact;
            std::array<std::unique_ptr<DominanceCount>, 3> dc;
            std::array<std::unique_ptr<NestedLoop>, 3> nl;
            for (unsigned party = 0; party < 3; ++party) {
                nlohmann::json j = {{"inputs", nlohmann::json::object()}};
                const auto suffix = "_" + std::to_string(party);
                for (auto key : {"sid", "dir", "t", "a"}) {
                    j["inputs"][std::string(key) + suffix] = nlohmann::json::array();
                }
                for (std::size_t row = 0; row < K; ++row) {
                    const bool outgoing = (party * K + row) % 2;
                    ref::PlainRow r{42, 100 + row % 7, 50 + row % 5, outgoing};
                    plain.push_back(r);
                    j["inputs"]["sid" + suffix].push_back(42);
                    j["inputs"]["dir" + suffix].push_back(outgoing);
                    j["inputs"]["t" + suffix].push_back(r.time);
                    j["inputs"]["a" + suffix].push_back(static_cast<std::uint64_t>(r.amount));
                }
                if (party == 2) {
                    j["inputs"]["subject"] = {42};
                }
                const auto file = directory / ("input" + std::to_string(party) + ".json");
                {
                    std::ofstream f(file);
                    f << j;
                }
                stores[party] = std::make_unique<MaterialStore>(
                    plan, party, directory / "m" / ("party_" + std::to_string(party) + ".bin"));
                stores[party]->preload();
                inputs[party] = std::make_unique<InputManager>(*stores[party], plan, file);
                compact[party] = std::make_unique<Compact>(c, *stores[party], plan, *inputs[party]);
                dc[party] = std::make_unique<DominanceCount>(c,
                                                             *stores[party],
                                                             plan,
                                                             compact[party]->inbound(),
                                                             compact[party]->outbound(),
                                                             compact[party]->omega_in(),
                                                             compact[party]->omega_out());
                nl[party] = std::make_unique<NestedLoop>(c,
                                                         *stores[party],
                                                         plan,
                                                         compact[party]->inbound(),
                                                         compact[party]->outbound(),
                                                         compact[party]->omega_in(),
                                                         compact[party]->omega_out(),
                                                         "output.omega.nl");
                stores[party]->claim_execution();
                stores[party]->begin_online();
            }
            Config config;
            config.hosts = dc_hosts(3);
            config.threads = 2;
            config.timeout_ms = 15000;
            std::array<std::exception_ptr, 3> errors{};
            std::vector<std::thread> threads;
            for (unsigned party = 0; party < 3; ++party) {
                threads.emplace_back([&, party] {
                    try {
                        omp_set_num_threads(2);
                        Net net(party, config);
                        inputs[party]->run(net);
                        compact[party]->run(net);
                        dc[party]->run(net);
                        nl[party]->run(net);
                    } catch (...) {
                        errors[party] = std::current_exception();
                    }
                });
            }
            for (auto &t : threads) {
                t.join();
            }
            for (auto e : errors) {
                if (e) {
                    std::rethrow_exception(e);
                }
            }
            std::uint64_t dc_count = 0, nl_count = 0, dc_omega = 0, nl_omega = 0;
            for (unsigned party = 0; party < 3; ++party) {
                dc_count += dc[party]->count_shares()[0];
                nl_count += nl[party]->count_shares()[0];
                dc_omega += dc[party]->overflow_share();
                nl_omega += nl[party]->overflow_share();
            }
            const auto ideal = ref::investigation(plain, 42, c);
            EXPECT_EQ(dc_count, ideal.counts[0]) << K;
            EXPECT_EQ(dc_count, nl_count) << K;
            EXPECT_EQ(dc_omega, nl_omega);
            EXPECT_EQ(dc_omega, ideal.overflow);
            EXPECT_GT(dc_count, 0u);
        }
    }
}
