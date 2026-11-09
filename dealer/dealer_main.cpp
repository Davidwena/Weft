#include <omp.h>
#include <vector>
#include <string>

namespace bj {
    extern std::vector<std::pair<std::string, double>> x11_generation_profile;
}

#include "bj/core/material.hpp"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cerrno>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char **argv) {
    try {
        bj::install_dealer_network_guard();

        errno = 0;
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            ::close(fd);
            throw std::runtime_error("dealer network guard failed");
        }
        if (errno != EPERM) {
            throw std::runtime_error("dealer network guard unexpected errno");
        }
        if (argc == 2 && std::string(argv[1]) == "--probe-network") {
            std::puts("{\"network_guard_active\":true,\"socket_errno\":1}");
            return 0;
        }
        std::string plan_path, out, mode = "random";
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) {
                throw std::invalid_argument("dealer options require values");
            }
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--plan") {
                plan_path = value;
            } else if (key == "--out") {
                out = value;
            } else if (key == "--mode") {
                mode = value;
            } else {
                throw std::invalid_argument("unknown dealer option");
            }
        }
        if (plan_path.empty() || out.empty()) {
            throw std::invalid_argument(
                "dealer_main --plan plan.json --out fresh_directory --mode random|prg");
        }
        omp_set_dynamic(0);
        omp_set_num_threads(16);
        const auto plan = bj::MaterialPlan::load(plan_path);
        const auto report = bj::generate_materials(plan, bj::parse_material_mode(mode), out);
        auto result = nlohmann::json{
            {"generation_workers", 16},
            {"generation_strategy", "byte-aligned offline kernel tiles; one global shuffle"},
            {"mode", mode},
            {"generation_seconds", report.seconds},
            {"network_guard_active", true},
            {"online_messages", 0},
            {"file_bytes_per_party", report.file_bytes},
            {"logical_share_bits_per_party", report.share_bits},
            {"gate_bits_per_party", report.gate_bits},
            {"dense_share_bytes_per_party", report.dense_share_bytes},
            {"private_owner_mask_bytes_per_party", report.owner_bytes}};
        result["generation_profile"] = bj::x11_generation_profile;
        std::puts(result.dump().c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "dealer_main: %s\n", e.what());
        return 1;
    }
}
