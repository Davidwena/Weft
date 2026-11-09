#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace bj {
    enum class OpenMode {
        Auto,
        Broadcast
    };
    OpenMode parse_open_mode(const std::string &mode);
    const char *open_mode_name(OpenMode mode) noexcept;

    struct Host {
        std::string address;
        std::uint16_t port;
    };

    struct Config {
        std::vector<Host> hosts;
        unsigned threads = 16;
        unsigned king = 0;
        OpenMode open_mode = OpenMode::Auto;
        unsigned timeout_ms = 30000;
        static Config load(const std::string &file);
        void validate(unsigned n) const;
    };
}
