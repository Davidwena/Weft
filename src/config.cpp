#include "bj/common/config.hpp"
#include <fstream>
#include <set>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace bj {
    OpenMode parse_open_mode(const std::string &mode) {
        if (mode == "auto") {
            return OpenMode::Auto;
        }
        if (mode == "broadcast") {
            return OpenMode::Broadcast;
        }
        throw std::invalid_argument("open mode must be auto or broadcast");
    }

    const char *open_mode_name(OpenMode mode) noexcept {
        return mode == OpenMode::Auto ? "auto" : "broadcast";
    }

    Config Config::load(const std::string &file) {
        std::ifstream stream(file);
        if (!stream) {
            throw std::runtime_error("cannot open config: " + file);
        }
        const auto j = nlohmann::json::parse(stream);
        Config c;
        auto positive = [](const nlohmann::json &v, std::int64_t max) {
            if (!v.is_number_integer()) {
                throw std::invalid_argument("integer config required");
            }
            const auto x = v.get<std::int64_t>();
            if (x <= 0 || x > max) {
                throw std::invalid_argument("config integer out of range");
            }
            return static_cast<unsigned>(x);
        };
        c.open_mode = parse_open_mode(j.value("open_mode", std::string("auto")));
        const auto king = j.value("king", nlohmann::json(0));
        if (!king.is_number_integer() || king.get<std::int64_t>() < 0 ||
            king.get<std::uint64_t>() > UINT32_MAX) {
            throw std::invalid_argument("invalid king id");
        }
        c.king = king.get<unsigned>();
        c.threads = positive(j.value("threads", nlohmann::json(16)), 1024);
        c.timeout_ms = positive(j.value("timeout_ms", nlohmann::json(30000)), 3600000);
        for (const auto &h : j.at("hosts")) {
            c.hosts.push_back({h.at("host").get<std::string>(),
                               static_cast<std::uint16_t>(positive(h.at("port"), 65535))});
        }
        c.validate(static_cast<unsigned>(c.hosts.size()));
        return c;
    }

    void Config::validate(unsigned n) const {
        if (n < 2 || n > 1024 || hosts.size() != n || threads == 0 || timeout_ms == 0 ||
            king >= n || (open_mode != OpenMode::Auto && open_mode != OpenMode::Broadcast)) {
            throw std::invalid_argument("invalid network config");
        }
        std::set<std::pair<std::string, std::uint16_t>> endpoints;
        for (const auto &h : hosts) {
            if (h.address.empty() || !h.port || !endpoints.emplace(h.address, h.port).second) {
                throw std::invalid_argument("invalid/duplicate endpoint");
            }
        }
    }
}
