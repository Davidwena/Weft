#include "../bench/net_driver.hpp"
#include "bj/core/party.hpp"
#include <cstring>

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--plan") == 0 || std::strncmp(argv[i], "--plan=", 7) == 0) {
            return bj::protocol_party_entry(argc, argv);
        }
    }
    try {
        auto o = bj::net_options(argc, argv);
        o.iterations = 1;
        o.warmup = 0;
        std::puts(bj::network_run(o).dump().c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "party_main: %s\n", e.what());
        return 1;
    }
}
