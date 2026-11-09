#include <set>
#include "bj/core/material.hpp"
#include "bj/core/masked_material.hpp"
#include "bj/plan/investigation.hpp"
#include "bj/proto/eda_shape.hpp"
#include "bj/core/comparison_dealer.hpp"
#include "bj/common/config.hpp"
#include "bj/common/timer.hpp"
#include <nlohmann/json.hpp>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <set>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bj {
    namespace {
        using Json = nlohmann::json;
        constexpr std::size_t header_bytes = 256, record_bytes = 64;
        constexpr char magic[9] = "BJMAT002";

        bool durable_material() {
            const char *setting = std::getenv("BJ_MATERIAL_DURABILITY");
            if (!setting || std::strcmp(setting, "durable") == 0) {
                return true;
            }
            if (std::strcmp(setting, "nondurable") == 0) {
                return false;
            }
            throw std::invalid_argument("BJ_MATERIAL_DURABILITY must be durable or nondurable");
        }

        std::uint64_t hash_bytes(const std::string &text) {
            std::uint64_t h = 14695981039346656037ULL;
            for (unsigned char c : text) {
                h ^= c;
                h *= 1099511628211ULL;
            }
            return h;
        }

        std::size_t integer(const Json &value, std::size_t max = SIZE_MAX) {
            if (!value.is_number_integer() ||
                (value.is_number_integer() && value.get<std::int64_t>() < 0)) {
                throw std::invalid_argument("expected nonnegative integer");
            }
            const auto x = value.get<std::uint64_t>();
            if (x > max) {
                throw std::invalid_argument("plan integer out of range");
            }
            return static_cast<std::size_t>(x);
        }

        void random_fill(std::span<std::byte> out) {
            while (!out.empty()) {
                const auto n =
                    ::getrandom(out.data(), std::min<std::size_t>(out.size(), 1 << 20), 0);
                if (n < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    throw std::runtime_error("getrandom failed");
                }
                if (!n) {
                    throw std::runtime_error("getrandom returned zero");
                }
                out = out.subspan(static_cast<std::size_t>(n));
            }
        }

        void trim(std::span<std::byte> data, std::uint64_t bits) {
            if (bits % 8) {
                data.back() &= std::byte((1u << (bits % 8)) - 1);
            }
        }

        std::size_t align64(std::size_t n) {
            if (n > SIZE_MAX - 63) {
                throw std::length_error("material offset overflow");
            }
            return (n + 63) & ~std::size_t(63);
        }

        void put64(std::span<std::byte> data, std::size_t offset, std::uint64_t value) {
            std::memcpy(data.data() + offset, &value, 8);
        }

        void put32(std::span<std::byte> data, std::size_t offset, unsigned value) {
            std::uint32_t v = value;
            std::memcpy(data.data() + offset, &v, 4);
        }

        std::uint64_t get64(std::span<const std::byte> data, std::size_t offset) {
            std::uint64_t x;
            std::memcpy(&x, data.data() + offset, 8);
            return x;
        }

        unsigned get32(std::span<const std::byte> data, std::size_t offset) {
            std::uint32_t x;
            std::memcpy(&x, data.data() + offset, 4);
            return x;
        }

        void encode_record(const Segment &s, std::span<std::byte> out) {
            std::fill(out.begin(), out.end(), std::byte{0});
            put64(out, 0, s.id);
            put64(out, 8, s.count);
            put64(out, 16, s.bits);
            put64(out, 24, s.offset);
            put64(out, 32, s.bytes);
            put64(out, 40, s.name_hash);
            put32(out, 48, s.width);
            put32(out, 52, static_cast<unsigned>(s.role));
            put32(out, 56, static_cast<unsigned>(s.storage));
        }

        struct Mapping {
            int fd = -1;
            std::size_t size = 0;
            std::byte *data = nullptr;

            Mapping(const std::filesystem::path &file, std::size_t bytes, bool write) {
                fd = ::open(file.c_str(),
                            write ? (O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW)
                                  : (O_RDONLY | O_CLOEXEC | O_NOFOLLOW),
                            0600);
                if (fd < 0) {
                    throw std::runtime_error("cannot open material: " + file.string());
                }
                try {
                    if (write && (bytes > static_cast<std::size_t>(INT64_MAX) ||
                                  ::ftruncate(fd, static_cast<off_t>(bytes)))) {
                        throw std::runtime_error("material truncate failed");
                    }

                    struct stat st {};

                    if (::fstat(fd, &st) || !S_ISREG(st.st_mode) ||
                        st.st_size < static_cast<off_t>(header_bytes)) {
                        throw std::runtime_error("invalid material file");
                    }
                    size = static_cast<std::size_t>(st.st_size);
                    auto *p = ::mmap(nullptr,
                                     size,
                                     write ? (PROT_READ | PROT_WRITE) : PROT_READ,
                                     write ? MAP_SHARED : MAP_PRIVATE,
                                     fd,
                                     0);
                    if (p == MAP_FAILED) {
                        throw std::runtime_error("material mmap failed");
                    }
                    data = static_cast<std::byte *>(p);
                } catch (...) {
                    ::close(fd);
                    fd = -1;
                    throw;
                }
            }

            ~Mapping() {
                if (data) {
                    ::munmap(data, size);
                }
                if (fd >= 0) {
                    ::close(fd);
                }
            }

            Mapping(const Mapping &) = delete;

            std::span<std::byte> span() noexcept {
                return {data, size};
            }

            std::span<const std::byte> span() const noexcept {
                return {data, size};
            }

            void flush() {
                if (durable_material() && (::msync(data, size, MS_SYNC) || ::fsync(fd))) {
                    throw std::runtime_error("material flush failed");
                }
            }
        };
    }

    MaterialMode parse_material_mode(const std::string &value) {
        if (value == "random") {
            return MaterialMode::Random;
        }
        if (value == "prg") {
            return MaterialMode::Prg;
        }
        throw std::invalid_argument("material mode must be random or prg");
    }

    const char *material_mode_name(MaterialMode mode) noexcept {
        return mode == MaterialMode::Random ? "random" : "prg";
    }

    MaterialPlan MaterialPlan::load(const std::filesystem::path &path) {
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("cannot open material plan");
        }
        const auto j = Json::parse(f);
        if (j.contains("investigation")) {
            return compile_investigation(InvestigationConfig::parse(j));
        }
        MaterialPlan p;
        if (j.value("version", 1) != 1) {
            throw std::invalid_argument("unsupported plan version");
        }
        p.n = static_cast<unsigned>(integer(j.at("n"), 1024));
        p.king = static_cast<unsigned>(integer(j.value("king", Json(0)), 1023));
        p.output_party = static_cast<unsigned>(integer(j.value("output_party", Json(0)), 1023));
        p.open_mode = j.value("open_mode", std::string("auto"));
        p.output_mode = j.value("output_mode", std::string("values"));
        for (const auto &v : j.at("inputs")) {
            InputSpec in;
            in.name = v.at("name").get<std::string>();
            in.owner = static_cast<unsigned>(integer(v.at("owner"), 1023));
            const auto type = v.value("type", std::string("bits"));
            if (type != "bits" && type != "ring") {
                throw std::invalid_argument("input type must be bits or ring");
            }
            in.width = type == "bits" ? 0 : static_cast<unsigned>(integer(v.at("k"), 96));
            if (type == "ring" && !in.width) {
                throw std::invalid_argument("ring width must be positive");
            }
            in.count = integer(v.at("count"), std::size_t{1} << 30);
            p.inputs.push_back(in);
        }
        p.context = j.value("context", std::string{});
        for (const auto &v : j.value("masks", Json::array())) {
            MaskSpec m;
            m.name = v.at("name").get<std::string>();
            m.width = static_cast<unsigned>(integer(v.at("k"), 96));
            m.count = integer(v.at("count"));
            m.fresh = v.at("fresh");
            m.distribute = v.at("distribute");
            for (const auto &t : v.at("terms")) {
                m.terms.push_back({integer(t.at("source")),
                                   integer(t.at("source_offset")),
                                   integer(t.at("destination_offset")),
                                   integer(t.at("length")),
                                   integer(t.at("divisor")),
                                   integer(t.at("modulus")),
                                   t.at("coefficient").get<std::int64_t>()});
            }
            p.masks.push_back(std::move(m));
        }
        for (const auto &v : j.value("bitmuls", Json::array())) {
            p.bitmuls.push_back(
                {v.at("name"), v.at("k"), integer(v.at("count")), integer(v.at("mask"))});
        }
        for (const auto &v : j.value("multiplications", Json::array())) {
            p.multiplications.push_back({v.at("name"), v.at("k"), integer(v.at("count"))});
        }
        for (const auto &v : j.value("shuffles", Json::array())) {
            ShuffleSpec q;
            q.name = v.at("name");
            q.count = integer(v.at("count"));
            q.inputs = v.at("inputs").get<std::vector<std::size_t>>();
            q.outputs = v.at("outputs").get<std::vector<std::size_t>>();
            p.shuffles.push_back(std::move(q));
        }
        for (const auto &v : j.value("gates", Json::array())) {
            GateSpec g;
            g.name = v.at("name").get<std::string>();
            g.m = static_cast<unsigned>(integer(v.at("m"), 4));
            g.width = static_cast<unsigned>(integer(v.at("k"), 96));
            g.count = integer(v.at("count"), std::size_t{1} << 30);
            for (const auto &name : v.at("inputs")) {
                auto it = std::find_if(p.inputs.begin(), p.inputs.end(), [&](const InputSpec &in) {
                    return in.name == name.get<std::string>();
                });
                if (it == p.inputs.end()) {
                    throw std::invalid_argument("unknown gate input");
                }
                g.inputs.push_back(static_cast<std::size_t>(it - p.inputs.begin()));
            }
            p.gates.push_back(std::move(g));
        }
        for (const auto &v : j.value("comparisons", Json::array())) {
            CompareSpec c;
            c.name = v.at("name").get<std::string>();
            c.kind = parse_compare_kind(v.at("kind").get<std::string>());
            c.width = static_cast<unsigned>(integer(v.at("k"), 96));
            c.d = static_cast<unsigned>(integer(v.value("d", Json(0)), 9));
            c.table_source = v.value("table_source", std::string{});
            c.evaluation_sources = v.value("evaluation_sources", std::vector<unsigned>{});
            c.evaluations = static_cast<unsigned>(integer(v.value("evaluations", Json(1)), 1024));
            c.count = integer(v.at("count"), std::size_t{1} << 30);
            for (const auto &name : v.at("inputs")) {
                auto it = std::find_if(p.inputs.begin(), p.inputs.end(), [&](const InputSpec &in) {
                    return in.name == name.get<std::string>();
                });
                if (it == p.inputs.end()) {
                    auto m =
                        std::find_if(p.masks.begin(), p.masks.end(), [&](const MaskSpec &spec) {
                            return spec.name == name.get<std::string>();
                        });
                    if (m == p.masks.end()) {
                        throw std::invalid_argument("unknown comparison input");
                    }
                    c.inputs.push_back(p.inputs.size() +
                                       static_cast<std::size_t>(m - p.masks.begin()));
                } else {
                    c.inputs.push_back(static_cast<std::size_t>(it - p.inputs.begin()));
                }
            }
            p.comparisons.push_back(std::move(c));
        }
        p.validate();
        return p;
    }

    void MaterialPlan::validate() const {
        if (n < 2 || n > 1024 || king >= n || output_party >= n ||
            (output_mode != "values" && output_mode != "sum")) {
            throw std::invalid_argument("invalid material plan");
        }
        parse_open_mode(open_mode);
        std::set<std::string> names;
        for (const auto &in : inputs) {
            if (in.name.empty() || !names.insert(in.name).second || in.owner >= n ||
                in.width > 96 || in.count > (std::size_t{1} << 30)) {
                throw std::invalid_argument("invalid input spec");
            }
        }
        for (std::size_t i = 0; i < masks.size(); ++i) {
            const auto &m = masks[i];
            if (m.name.empty() || !names.insert(m.name).second || m.width > 96 ||
                m.count > (std::size_t{1} << 30) || (m.fresh && !m.terms.empty()) ||
                (!m.fresh && m.distribute)) {
                throw std::invalid_argument("invalid mask spec");
            }
            for (const auto &t : m.terms) {
                if (t.source >= inputs.size() + i || !t.divisor || !t.modulus ||
                    t.destination_offset > m.count || t.length > m.count - t.destination_offset) {
                    throw std::invalid_argument("invalid mask wiring");
                }
                const auto count = t.source < inputs.size() ? inputs[t.source].count
                                                            : masks[t.source - inputs.size()].count;
                const auto width = t.source < inputs.size() ? inputs[t.source].width
                                                            : masks[t.source - inputs.size()].width;
                if ((!width) != (!m.width) || (m.width > width && width != 0) ||
                    t.source_offset > count ||
                    (t.length && std::min(t.modulus - 1, (t.length - 1) / t.divisor) >=
                                     count - t.source_offset)) {
                    throw std::invalid_argument("mask wiring range/domain");
                }
                if (!m.width && t.coefficient != 1) {
                    throw std::invalid_argument("Boolean mask wiring is XOR");
                }
            }
        }
        for (const auto &q : shuffles) {
            if (q.name.empty() || !q.count || q.count > UINT32_MAX || q.inputs.empty() ||
                q.inputs.size() != q.outputs.size()) {
                throw std::invalid_argument("shuffle shape");
            }
            std::set<std::size_t> out;
            for (std::size_t c = 0; c < q.inputs.size(); ++c) {
                auto in = q.inputs[c], o = q.outputs[c];
                if (in >= inputs.size() + masks.size() || o < inputs.size() ||
                    o >= inputs.size() + masks.size() || !out.insert(o).second) {
                    throw std::invalid_argument("shuffle column binding");
                }
                const auto iw =
                    in < inputs.size() ? inputs[in].width : masks[in - inputs.size()].width;
                const auto ic =
                    in < inputs.size() ? inputs[in].count : masks[in - inputs.size()].count;
                const auto &om = masks[o - inputs.size()];
                if (!om.fresh || !om.distribute || ic != q.count || om.count != q.count ||
                    iw != om.width) {
                    throw std::invalid_argument(
                        "shuffle masks must be independently generated positional masks");
                }
            }
        }
        names.clear();
        for (const auto &g : gates) {
            if (g.name.empty() || !names.insert(g.name).second || !g.m || g.m > 4 || !g.width ||
                g.width > 96 ||
                (g.inputs.size() != g.m && !(g.inputs.empty() && !context.empty())) ||
                g.count > (std::size_t{1} << 30)) {
                throw std::invalid_argument("invalid gate spec");
            }
            for (auto id : g.inputs) {
                if (id >= inputs.size() || inputs[id].width != 0 || inputs[id].count != g.count) {
                    throw std::invalid_argument("gate input shape/type mismatch");
                }
            }
        }
        for (const auto &c : comparisons) {
            if (c.name.empty() || !names.insert(c.name).second || !c.width || c.width > 96 ||
                c.d > 9 || !c.evaluations || c.evaluations > 1024 ||
                c.count > (std::size_t{1} << 30) || static_cast<unsigned>(c.kind) > 10) {
                throw std::invalid_argument("invalid comparison spec");
            }
            if ((c.kind == CompareKind::Interval || c.kind == CompareKind::IntervalFresh) &&
                c.width < 2) {
                throw std::invalid_argument("interval requires at least 2 bits");
            }
            const bool lts = c.kind == CompareKind::LtsKnown || c.kind == CompareKind::LtsFresh;
            const bool msb = c.kind == CompareKind::MsbKnown || c.kind == CompareKind::MsbFresh ||
                             c.kind == CompareKind::Interval ||
                             c.kind == CompareKind::IntervalFresh;
            const bool detached =
                !context.empty() && c.inputs.empty() &&
                (c.kind == CompareKind::MsbFresh || c.kind == CompareKind::IntervalFresh ||
                 c.kind == CompareKind::EqFresh);
            if (!detached && c.inputs.size() != (lts ? 2 : 1)) {
                throw std::invalid_argument("comparison input arity");
            }
            for (auto id : c.inputs) {
                if (id >= inputs.size() + masks.size() ||
                    (id < inputs.size() ? inputs[id].width : masks[id - inputs.size()].width) !=
                        c.width ||
                    (id < inputs.size() ? inputs[id].count : masks[id - inputs.size()].count) !=
                        c.count) {
                    throw std::invalid_argument("comparison input shape/type");
                }
            }
            if (c.kind == CompareKind::LtsKnown && c.inputs[0] == c.inputs[1]) {
                throw std::invalid_argument("stored LTS requires independently masked columns");
            }
            if (lts && c.evaluations != 1) {
                throw std::invalid_argument("LTS evaluations must be 1");
            }
            if ((c.kind == CompareKind::MsbFresh || c.kind == CompareKind::EqFresh) &&
                c.evaluations != 1) {
                throw std::invalid_argument("fresh MSB evaluations must be 1");
            }
            if (!c.evaluation_sources.empty() && c.table_source.empty()) {
                throw std::invalid_argument("evaluation alias requires a shared root");
            }
            if (!c.table_source.empty()) {
                const auto root =
                    std::find_if(comparisons.begin(), comparisons.end(), [&](const auto &r) {
                        return r.name == c.table_source;
                    });
                if (root == comparisons.end() || &*root >= &c || !root->table_source.empty() ||
                    c.kind != CompareKind::Interval || root->kind != c.kind ||
                    c.width != root->width || c.count != root->count ||
                    c.block_width() != root->block_width() || c.inputs != root->inputs) {
                    throw std::invalid_argument("unsafe shared threshold table");
                }
                if (!c.evaluation_sources.empty()) {
                    std::set<unsigned> seen;
                    if (c.evaluation_sources.size() != c.evaluations) {
                        throw std::invalid_argument("evaluation alias shape");
                    }
                    for (auto e : c.evaluation_sources) {
                        if (e >= root->evaluations || !seen.insert(e).second) {
                            throw std::invalid_argument("evaluation alias range/reuse");
                        }
                    }
                }
            }
            if (c.kind == CompareKind::EdaMsb || c.kind == CompareKind::EdaEq) {
                if (c.evaluations != 1 || c.d) {
                    throw std::invalid_argument("baseline does not reuse material or block tables");
                }
                continue;
            }
            const auto ell = msb ? c.width - 1 : c.width;
            if (ell) {
                CompareShape shape(ell,
                                   c.block_width(),
                                   (c.kind == CompareKind::Eq || c.kind == CompareKind::EqFresh));
            }
        }
        for (const auto &b : bitmuls) {
            if (b.name.empty() || !names.insert(b.name).second || !b.width || b.width > 96 ||
                b.count > (std::size_t{1} << 30) || b.mask >= inputs.size() + masks.size()) {
                throw std::invalid_argument("invalid bitmul spec");
            }
            const auto w =
                b.mask < inputs.size() ? inputs[b.mask].width : masks[b.mask - inputs.size()].width;
            const auto count =
                b.mask < inputs.size() ? inputs[b.mask].count : masks[b.mask - inputs.size()].count;
            if (w < b.width || count != b.count) {
                throw std::invalid_argument("bitmul mask shape/domain");
            }
        }
        for (const auto &m : multiplications) {
            if (m.name.empty() || !names.insert(m.name).second ||
                (m.width != 32 && m.width != 64) || m.count > (std::size_t{1} << 30)) {
                throw std::invalid_argument("invalid multiplication spec");
            }
        }
        if (gates.empty() && comparisons.empty() && shuffles.empty() && multiplications.empty() &&
            bitmuls.empty()) {
            throw std::invalid_argument("plan must contain a protocol");
        }
    }

    std::string MaterialPlan::canonical() const {
        Json j = {{"version", 1},
                  {"n", n},
                  {"king", king},
                  {"output_party", output_party},
                  {"open_mode", open_mode},
                  {"output_mode", output_mode},
                  {"inputs", Json::array()},
                  {"gates", Json::array()}};
        for (const auto &in : inputs) {
            Json v = {{"name", in.name},
                      {"owner", in.owner},
                      {"count", in.count},
                      {"type", in.width ? "ring" : "bits"}};
            if (in.width) {
                v["k"] = in.width;
            }
            j["inputs"].push_back(v);
        }
        for (const auto &g : gates) {
            Json names = Json::array();
            for (auto id : g.inputs) {
                names.push_back(inputs.at(id).name);
            }
            j["gates"].push_back({{"name", g.name},
                                  {"m", g.m},
                                  {"k", g.width},
                                  {"count", g.count},
                                  {"inputs", names}});
        }
        if (!comparisons.empty()) {
            j["comparisons"] = Json::array();
            for (const auto &c : comparisons) {
                Json names = Json::array();
                for (auto id : c.inputs) {
                    names.push_back(id < inputs.size() ? inputs[id].name
                                                       : masks.at(id - inputs.size()).name);
                }
                j["comparisons"].push_back({{"name", c.name},
                                            {"kind", compare_kind_name(c.kind)},
                                            {"k", c.width},
                                            {"d", c.d},
                                            {"count", c.count},
                                            {"evaluations", c.evaluations},
                                            {"inputs", names}});
            }
        }
        if (!context.empty()) {
            j["context"] = context;
        }
        if (!masks.empty()) {
            j["masks"] = Json::array();
            for (const auto &m : masks) {
                Json terms = Json::array();
                for (const auto &t : m.terms) {
                    terms.push_back({{"source", t.source},
                                     {"source_offset", t.source_offset},
                                     {"destination_offset", t.destination_offset},
                                     {"length", t.length},
                                     {"divisor", t.divisor},
                                     {"modulus", t.modulus},
                                     {"coefficient", t.coefficient}});
                }
                j["masks"].push_back({{"name", m.name},
                                      {"k", m.width},
                                      {"count", m.count},
                                      {"fresh", m.fresh},
                                      {"distribute", m.distribute},
                                      {"terms", terms}});
            }
        }
        if (!bitmuls.empty()) {
            j["bitmuls"] = Json::array();
            for (const auto &b : bitmuls) {
                j["bitmuls"].push_back(
                    {{"name", b.name}, {"k", b.width}, {"count", b.count}, {"mask", b.mask}});
            }
        }
        for (std::size_t i = 0; i < comparisons.size(); ++i) {
            if (!comparisons[i].table_source.empty()) {
                j["comparisons"][i]["table_source"] = comparisons[i].table_source;
            }
        }
        for (std::size_t i = 0; i < comparisons.size(); ++i) {
            if (!comparisons[i].evaluation_sources.empty()) {
                j["comparisons"][i]["evaluation_sources"] = comparisons[i].evaluation_sources;
            }
        }
        if (!multiplications.empty()) {
            j["multiplications"] = Json::array();
            for (const auto &m : multiplications) {
                j["multiplications"].push_back(
                    {{"name", m.name}, {"k", m.width}, {"count", m.count}});
            }
        }
        if (!shuffles.empty()) {
            j["shuffles"] = Json::array();
            for (const auto &q : shuffles) {
                j["shuffles"].push_back({{"name", q.name},
                                         {"count", q.count},
                                         {"inputs", q.inputs},
                                         {"outputs", q.outputs}});
            }
        }
        return j.dump();
    }

    std::uint64_t MaterialPlan::fingerprint() const {
        return hash_bytes(canonical());
    }

    MaterialLayout::MaterialLayout(const MaterialPlan &p, unsigned party, MaterialMode mode) {
        p.validate();
        if (party >= p.n || (mode != MaterialMode::Random && mode != MaterialMode::Prg)) {
            throw std::invalid_argument("invalid material role");
        }
        auto add = [&](SegmentRole role,
                       std::size_t index,
                       unsigned width,
                       std::size_t count,
                       std::size_t subset,
                       std::string name,
                       unsigned owner = 0) {
            Segment s;
            s.id = segments.size();
            s.index = index;
            s.subset = subset;
            s.width = width;
            s.count = count;
            s.role = role;
            s.bits = checked_bits(count, width ? width : 1);
            s.bytes = bits_bytes(s.bits);
            s.name_hash = hash_bytes(name);
            s.storage =
                (role == SegmentRole::OwnerMask || role == SegmentRole::ShufflePrivate)
                    ? (party == owner ? SegmentStorage::Dense : SegmentStorage::Absent)
                    : ((mode == MaterialMode::Prg && party + 1 < p.n) ? SegmentStorage::Seeded
                                                                      : SegmentStorage::Dense);
            segments.push_back(s);
            return s.id;
        };
        for (std::size_t i = 0; i < p.inputs.size(); ++i) {
            const auto &in = p.inputs[i];
            input_shares.push_back(add(
                SegmentRole::InputShare, i, in.width, in.count, 0, "input." + in.name + ".share"));
            owner_masks.push_back(add(SegmentRole::OwnerMask,
                                      i,
                                      in.width,
                                      in.count,
                                      0,
                                      "input." + in.name + ".owner",
                                      in.owner));
        }
        for (std::size_t i = 0; i < p.masks.size(); ++i) {
            const auto &m = p.masks[i];
            masks.push_back(
                m.distribute ? add(SegmentRole::MaskShare, i, m.width, m.count, 0, "mask." + m.name)
                             : SIZE_MAX);
        }
        for (std::size_t i = 0; i < p.shuffles.size(); ++i) {
            const auto &q = p.shuffles[i];
            ShuffleSegments ids;
            for (unsigned owner = 0; owner < p.n; ++owner) {
                ids.permutations.push_back(
                    add(SegmentRole::ShufflePrivate,
                        i,
                        mode == MaterialMode::Prg ? 0 : 32,
                        mode == MaterialMode::Prg ? 128 : q.count,
                        owner,
                        "shuffle." + q.name + ".perm." + std::to_string(owner),
                        owner));
                ids.corrections.emplace_back();
                for (std::size_t c = 0; c < q.inputs.size(); ++c) {
                    const auto col = q.inputs[c];
                    const auto width = col < p.inputs.size() ? p.inputs[col].width
                                                             : p.masks[col - p.inputs.size()].width;
                    ids.corrections.back().push_back(add(SegmentRole::ShufflePrivate,
                                                         i,
                                                         width,
                                                         q.count,
                                                         c,
                                                         "shuffle." + q.name + ".corr." +
                                                             std::to_string(owner) + "." +
                                                             std::to_string(c),
                                                         owner));
                }
            }
            shuffles.push_back(std::move(ids));
        }
        for (std::size_t i = 0; i < p.gates.size(); ++i) {
            const auto &g = p.gates[i];
            gate_masks.push_back(add(SegmentRole::GateMask,
                                     i,
                                     0,
                                     checked_bits(g.count, g.m),
                                     0,
                                     "gate." + g.name + ".tau"));
            gate_products.emplace_back();
            for (unsigned subset = 1; subset < (1u << g.m); ++subset) {
                gate_products.back().push_back(
                    add(SegmentRole::GateProduct,
                        i,
                        g.width,
                        g.count,
                        subset,
                        "gate." + g.name + ".product." + std::to_string(subset)));
            }
        }
        for (std::size_t i = 0; i < p.bitmuls.size(); ++i) {
            const auto &b = p.bitmuls[i];
            std::array<std::size_t, 3> ids{};
            for (unsigned j = 0; j < 3; ++j) {
                ids[j] = add(SegmentRole::BitMul,
                             i,
                             j ? b.width : 0,
                             b.count,
                             j,
                             "bitmul." + b.name + "." + std::to_string(j));
            }
            bitmuls.push_back(ids);
        }
        for (std::size_t i = 0; i < p.multiplications.size(); ++i) {
            const auto &m = p.multiplications[i];
            std::array<std::size_t, 3> ids{};
            for (unsigned j = 0; j < 3; ++j) {
                ids[j] = add(SegmentRole::Multiply,
                             i,
                             m.width,
                             m.count,
                             j,
                             "multiply." + m.name + "." + std::to_string(j));
            }
            multiplications.push_back(ids);
        }
        for (std::size_t i = 0; i < p.comparisons.size(); ++i) {
            const auto &c = p.comparisons[i];
            ComparisonSegments ids;
            if ((c.kind == CompareKind::Interval || c.kind == CompareKind::IntervalFresh) &&
                c.width < 2) {
                throw std::invalid_argument("interval requires at least 2 bits");
            }
            const bool lts = c.kind == CompareKind::LtsKnown || c.kind == CompareKind::LtsFresh;
            const bool msb = c.kind == CompareKind::MsbKnown || c.kind == CompareKind::MsbFresh ||
                             c.kind == CompareKind::Interval ||
                             c.kind == CompareKind::IntervalFresh;
            const bool fresh = c.kind == CompareKind::MsbFresh || c.kind == CompareKind::LtsFresh ||
                               c.kind == CompareKind::IntervalFresh ||
                               c.kind == CompareKind::EqFresh;
            if (c.kind == CompareKind::EdaMsb || c.kind == CompareKind::EdaEq) {
                EdaShape shape(c.width, c.kind == CompareKind::EdaEq);
                ids.aux.push_back(
                    add(SegmentRole::CompareAux, i, c.width, c.count, 0, "eda." + c.name + ".r"));
                ids.aux.push_back(add(SegmentRole::CompareAux,
                                      i,
                                      0,
                                      c.width * c.count,
                                      1,
                                      "eda." + c.name + ".bits"));
                ids.evaluations.push_back({add(SegmentRole::CompareEval,
                                               i,
                                               0,
                                               3 * shape.gates * c.count,
                                               0,
                                               "eda." + c.name + ".triples")});
                comparisons.push_back(std::move(ids));
                continue;
            }
            const auto ell = msb ? c.width - 1 : c.width;
            if (ell) {
                CompareShape shape(ell,
                                   c.block_width(),
                                   (c.kind == CompareKind::Eq || c.kind == CompareKind::EqFresh));
                if (!c.table_source.empty()) {
                    const auto it =
                        std::find_if(p.comparisons.begin(),
                                     p.comparisons.begin() + static_cast<std::ptrdiff_t>(i),
                                     [&](const auto &x) {
                                         return x.name == c.table_source;
                                     });
                    ids.tables =
                        comparisons.at(static_cast<std::size_t>(it - p.comparisons.begin())).tables;
                } else {
                    for (unsigned t = 0; t < (lts ? 3u : 1u); ++t) {
                        ids.tables.push_back(
                            add(SegmentRole::CompareTable,
                                i,
                                0,
                                checked_bits(c.count, 1) * shape.table_bits(),
                                t,
                                "compare." + c.name + ".table." + std::to_string(t)));
                    }
                }
                if (!c.evaluation_sources.empty()) {
                    const auto root =
                        std::find_if(p.comparisons.begin(),
                                     p.comparisons.begin() + static_cast<std::ptrdiff_t>(i),
                                     [&](const auto &x) {
                                         return x.name == c.table_source;
                                     });
                    const auto &root_ids =
                        comparisons.at(static_cast<std::size_t>(root - p.comparisons.begin()));
                    for (auto e : c.evaluation_sources) {
                        ids.evaluations.push_back(root_ids.evaluations.at(e));
                    }
                } else {
                    for (unsigned e = 0; e < c.evaluations; ++e) {
                        ids.evaluations.emplace_back();
                        for (unsigned t = 0; t < (lts ? 3u : 1u); ++t) {
                            ids.evaluations.back().push_back(add(SegmentRole::CompareEval,
                                                                 i,
                                                                 0,
                                                                 c.count * shape.eval_bits(),
                                                                 e,
                                                                 "compare." + c.name + ".eval." +
                                                                     std::to_string(e) + "." +
                                                                     std::to_string(t)));
                        }
                    }
                }
            } else {
                ids.evaluations.resize(c.evaluations);
            }
            if (fresh) {
                ids.aux.push_back(add(
                    SegmentRole::CompareAux, i, c.width, c.count, 0, "compare." + c.name + ".r"));
                if (lts) {
                    ids.aux.push_back(add(SegmentRole::CompareAux,
                                          i,
                                          c.width,
                                          c.count,
                                          1,
                                          "compare." + c.name + ".rp"));
                }
            }
            if ((msb && c.kind != CompareKind::Interval && c.kind != CompareKind::IntervalFresh) ||
                lts) {
                ids.aux.push_back(
                    add(SegmentRole::CompareAux, i, 0, c.count, 2, "compare." + c.name + ".sign"));
            }
            comparisons.push_back(std::move(ids));
        }
        if (segments.size() > (SIZE_MAX - header_bytes) / record_bytes) {
            throw std::length_error("material directory too big");
        }
        auto offset = align64(header_bytes + segments.size() * record_bytes);
        for (auto &s : segments) {
            if (s.role != SegmentRole::OwnerMask && s.role != SegmentRole::ShufflePrivate) {
                logical_share_bits += s.bits;
                if (s.role != SegmentRole::InputShare && s.role != SegmentRole::MaskShare) {
                    gate_bits += s.bits;
                }
            }
            if (s.storage == SegmentStorage::Dense) {
                s.offset = offset;
                if (s.bytes > SIZE_MAX - offset) {
                    throw std::length_error("material too large");
                }
                offset = align64(offset + s.bytes);
                if (s.role == SegmentRole::OwnerMask) {
                    owner_bytes += s.bytes;
                } else {
                    dense_share_bytes += s.bytes;
                }
            }
        }
        file_bytes = offset;
        seed_bytes = (mode == MaterialMode::Prg && party + 1 < p.n) ? 16 : 0;
    }

    DealerReport generate_materials(const MaterialPlan &p,
                                    MaterialMode mode,
                                    const std::filesystem::path &directory) {
        p.validate();
        Timer timer;
        if (std::filesystem::exists(directory) && !std::filesystem::is_empty(directory)) {
            throw std::runtime_error("dealer output directory must be new or empty");
        }
        std::filesystem::create_directories(directory);
        if (::chmod(directory.c_str(), 0700)) {
            throw std::runtime_error("cannot protect dealer directory");
        }
        DealerReport report;
        random_fill(report.run_id);
        std::vector<Prg::Key> seeds(p.n);
        std::vector<Prg> prgs;
        prgs.reserve(p.n);
        for (unsigned id = 0; id < p.n; ++id) {
            if (mode == MaterialMode::Prg && id + 1 < p.n) {
                random_fill(seeds[id]);
            }
            prgs.emplace_back(seeds[id]);
        }
        std::vector<MaterialLayout> layouts;
        std::vector<std::unique_ptr<Mapping>> files;
        layouts.reserve(p.n);
        files.reserve(p.n);
        for (unsigned id = 0; id < p.n; ++id) {
            layouts.emplace_back(p, id, mode);
            const auto &l = layouts.back();
            files.push_back(std::make_unique<Mapping>(
                directory / ("party_" + std::to_string(id) + ".bin"), l.file_bytes, true));
            auto data = files.back()->span();
            std::memcpy(data.data(), magic, 8);
            put64(data, 8, 2);
            put64(data, 16, p.fingerprint());
            put32(data, 24, id);
            put32(data, 28, p.n);
            put32(data, 32, static_cast<unsigned>(mode));
            put32(data, 36, static_cast<unsigned>(l.segments.size()));
            put64(data, 40, l.file_bytes);
            std::memcpy(data.data() + 48, report.run_id.data(), 16);
            std::memcpy(data.data() + 64, seeds[id].data(), 16);
            put64(data, 80, l.logical_share_bits);
            put64(data, 88, l.gate_bits);
            put64(data, 96, l.dense_share_bytes);
            put64(data, 104, l.owner_bytes);
            for (const auto &s : l.segments) {
                encode_record(s, data.subspan(header_bytes + s.id * record_bytes, record_bytes));
            }
            report.file_bytes.push_back(l.file_bytes);
            report.share_bits.push_back(l.logical_share_bits);
            report.gate_bits.push_back(l.gate_bits);
            report.dense_share_bytes.push_back(l.dense_share_bytes);
            report.owner_bytes.push_back(l.owner_bytes);
        }
        auto dense = [&](unsigned id, std::size_t segment) {
            const auto &s = layouts[id].segments[segment];
            if (s.storage != SegmentStorage::Dense) {
                throw std::logic_error("dense material destination");
            }
            return files[id]->span().subspan(s.offset, s.bytes);
        };
        auto random_share = [&](unsigned id, std::size_t segment, std::span<std::byte> out) {
            if (mode == MaterialMode::Prg) {
                prgs[id].fill(segment + 1, 0, out);
            } else {
                random_fill(out);
            }
            trim(out, layouts[id].segments[segment].bits);
        };
        auto split_bits = [&](std::size_t segment, std::span<const std::byte> secret) {
            AlignedBuffer<std::byte> temp(secret.size());
            auto last = dense(p.n - 1, segment);
            if (!secret.empty()) {
                std::memcpy(last.data(), secret.data(), secret.size());
            }
            for (unsigned id = 0; id + 1 < p.n; ++id) {
                auto share = mode == MaterialMode::Random ? dense(id, segment) : temp.span();
                random_share(id, segment, share);
                for (std::size_t i = 0; i < share.size(); ++i) {
                    last[i] ^= share[i];
                }
            }
        };
        auto split_ring = [&]<class T>(std::size_t segment, std::span<T> secret, unsigned k) {
            const auto bytes = bits_bytes(checked_bits(secret.size(), k));
            AlignedBuffer<std::byte> temp(bytes);
            AlignedBuffer<T> last(secret.size()), share_values(secret.size());
            std::copy(secret.begin(), secret.end(), last.span().begin());
            const auto mask = native_mask<T>(k);
            for (unsigned id = 0; id + 1 < p.n; ++id) {
                auto share = mode == MaterialMode::Random ? dense(id, segment) : temp.span();
                random_share(id, segment, share);
                unpack_native<T>(k, share, share_values.span());
                for (std::size_t i = 0; i < secret.size(); ++i) {
                    last.data()[i] = (last.data()[i] - share_values.data()[i]) & mask;
                }
            }
            pack_native<T>(k, last.span(), dense(p.n - 1, segment));
        };
        const auto &l = layouts[0];
        std::vector<AlignedBuffer<std::byte>> mask_values;
        mask_values.reserve(p.inputs.size() + p.masks.size());
        for (std::size_t index = 0; index < p.inputs.size(); ++index) {
            const auto &in = p.inputs[index];
            const auto segment = l.input_shares[index];
            AlignedBuffer<std::byte> secret(l.segments[segment].bytes);
            random_fill(secret.span());
            trim(secret.span(), l.segments[segment].bits);
            auto target = dense(in.owner, l.owner_masks[index]);
            if (!target.empty()) {
                std::memcpy(target.data(), secret.data(), target.size());
            }
            mask_values.emplace_back(secret.size());
            std::copy(
                secret.span().begin(), secret.span().end(), mask_values.back().span().begin());
            if (!in.width) {
                split_bits(segment, secret.span());
            } else if (in.width <= 32) {
                AlignedBuffer<std::uint32_t> native(in.count);
                unpack_native<std::uint32_t>(in.width, secret.span(), native.span());
                split_ring(segment, native.span(), in.width);
            } else if (in.width <= 64) {
                AlignedBuffer<std::uint64_t> native(in.count);
                unpack_native<std::uint64_t>(in.width, secret.span(), native.span());
                split_ring(segment, native.span(), in.width);
            } else {
                AlignedBuffer<uint128_t> native(in.count);
                unpack_native<uint128_t>(in.width, secret.span(), native.span());
                split_ring(segment, native.span(), in.width);
            }
        }
        make_mask_values(p, mask_values, random_fill);
        for (std::size_t i = 0; i < p.masks.size(); ++i) {
            if (p.masks[i].distribute) {
                const auto &m = p.masks[i];
                auto bytes = mask_values[p.inputs.size() + i].span();
                if (!m.width) {
                    split_bits(l.masks[i], bytes);
                } else if (m.width <= 32) {
                    AlignedBuffer<std::uint32_t> values(m.count);
                    unpack_native<std::uint32_t>(m.width, bytes, values.span());
                    split_ring(l.masks[i], values.span(), m.width);
                } else if (m.width <= 64) {
                    AlignedBuffer<std::uint64_t> values(m.count);
                    unpack_native<std::uint64_t>(m.width, bytes, values.span());
                    split_ring(l.masks[i], values.span(), m.width);
                } else {
                    AlignedBuffer<uint128_t> values(m.count);
                    unpack_native<uint128_t>(m.width, bytes, values.span());
                    split_ring(l.masks[i], values.span(), m.width);
                }
            }
        }
        for (std::size_t index = 0; index < p.gates.size(); ++index) {
            const auto &g = p.gates[index];
            const auto mask_segment = l.gate_masks[index];
            AlignedBuffer<std::byte> tau(l.segments[mask_segment].bytes);
            random_fill(tau.span());
            trim(tau.span(), l.segments[mask_segment].bits);
            split_bits(mask_segment, tau.span());
            auto products = [&]<class T>() {
                AlignedBuffer<T> values(g.count);
                for (unsigned subset = 1; subset < (1u << g.m); ++subset) {
                    for (std::size_t i = 0; i < g.count; ++i) {
                        unsigned bit = 1;
                        for (unsigned j = 0; j < g.m; ++j) {
                            if (subset & (1u << j)) {
                                bit &= static_cast<unsigned>(
                                    load_packed(tau.span(), j * g.count + i, 1));
                            }
                        }
                        values.data()[i] = static_cast<T>(bit);
                    }
                    split_ring(l.gate_products[index][subset - 1], values.span(), g.width);
                }
            };
            if (g.width <= 32) {
                products.template operator()<std::uint32_t>();
            } else if (g.width <= 64) {
                products.template operator()<std::uint64_t>();
            } else {
                products.template operator()<uint128_t>();
            }
        }
        for (std::size_t i = 0; i < p.bitmuls.size(); ++i) {
            const auto &b = p.bitmuls[i];
            AlignedBuffer<std::byte> tau(bits_bytes(b.count));
            random_fill(tau.span());
            trim(tau.span(), b.count);
            split_bits(l.bitmuls[i][0], tau.span());
            const auto width = b.mask < p.inputs.size() ? p.inputs[b.mask].width
                                                        : p.masks[b.mask - p.inputs.size()].width;
            auto generate = [&]<class T>() {
                AlignedBuffer<T> arithmetic(b.count), product(b.count);
                for (std::size_t j = 0; j < b.count; ++j) {
                    arithmetic.data()[j] = static_cast<T>(load_packed(tau.span(), j, 1));
                    product.data()[j] =
                        arithmetic.data()[j] *
                        static_cast<T>(load_wide(mask_values[b.mask].span(), j * width, width));
                }
                split_ring(l.bitmuls[i][1], arithmetic.span(), b.width);
                split_ring(l.bitmuls[i][2], product.span(), b.width);
            };
            if (b.width <= 32) {
                generate.template operator()<std::uint32_t>();
            } else if (b.width <= 64) {
                generate.template operator()<std::uint64_t>();
            } else {
                generate.template operator()<uint128_t>();
            }
        }
        for (std::size_t i = 0; i < p.multiplications.size(); ++i) {
            const auto &m = p.multiplications[i];
            auto generate = [&]<class T>() {
                AlignedBuffer<T> a(m.count), b(m.count), c(m.count);
                random_fill(std::as_writable_bytes(a.span()));
                random_fill(std::as_writable_bytes(b.span()));
                for (std::size_t j = 0; j < m.count; ++j) {
                    c.data()[j] = a.data()[j] * b.data()[j];
                }
                split_ring(l.multiplications[i][0], a.span(), m.width);
                split_ring(l.multiplications[i][1], b.span(), m.width);
                split_ring(l.multiplications[i][2], c.span(), m.width);
            };
            if (m.width == 32) {
                generate.template operator()<std::uint32_t>();
            } else {
                generate.template operator()<std::uint64_t>();
            }
        }
        {
            std::vector<std::span<const std::byte>> full_masks;
            for (const auto &mask : mask_values) {
                full_masks.push_back(mask.span());
            }
            make_shuffle_values(
                p,
                l,
                mode,
                full_masks,
                [&](std::size_t id, auto values) {
                    for (unsigned party = 0; party < p.n; ++party) {
                        if (layouts[party].segments[id].storage == SegmentStorage::Dense) {
                            auto target = dense(party, id);
                            if (target.size() != values.size()) {
                                throw std::logic_error("shuffle material shape");
                            }
                            std::copy(values.begin(), values.end(), target.begin());
                        }
                    }
                },
                random_fill);
            generate_comparison_material(
                p,
                l,
                full_masks,
                [&](std::size_t id, std::span<const std::byte> values) {
                    const auto &segment = l.segments[id];
                    if (!segment.width) {
                        split_bits(id, values);
                    } else if (segment.width <= 32) {
                        AlignedBuffer<std::uint32_t> v(segment.count);
                        unpack_native<std::uint32_t>(segment.width, values, v.span());
                        split_ring(id, v.span(), segment.width);
                    } else if (segment.width <= 64) {
                        AlignedBuffer<std::uint64_t> v(segment.count);
                        unpack_native<std::uint64_t>(segment.width, values, v.span());
                        split_ring(id, v.span(), segment.width);
                    } else {
                        AlignedBuffer<uint128_t> v(segment.count);
                        unpack_native<uint128_t>(segment.width, values, v.span());
                        split_ring(id, v.span(), segment.width);
                    }
                },
                random_fill);
        }
        for (auto &file : files) {
            file->flush();
        }
        {
            std::ofstream f(directory / "plan.json");
            f << p.canonical() << '\n';
            if (!f) {
                throw std::runtime_error("cannot persist public plan");
            }
        }
        {
            const auto ready = directory / "READY";
            int fd =
                ::open(ready.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
            if (fd < 0) {
                throw std::runtime_error("cannot publish completed material package");
            }
            std::array<std::byte, 24> marker{};
            std::memcpy(marker.data(), report.run_id.data(), 16);
            put64(marker, 16, p.fingerprint());
            const auto written = ::write(fd, marker.data(), marker.size());
            const auto synced = durable_material() ? ::fsync(fd) : 0;
            const auto closed = ::close(fd);
            const int dir = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            const bool directory_error = dir < 0 || (durable_material() && ::fsync(dir) != 0);
            if (dir >= 0) {
                ::close(dir);
            }
            if (written != static_cast<ssize_t>(marker.size()) || synced || closed ||
                directory_error) {
                throw std::runtime_error("cannot finish material package");
            }
        }
        report.seconds = timer.seconds();
        return report;
    }

    struct MaterialStore::Impl {
        MaterialPlan plan;
        unsigned party;
        MaterialMode mode;
        std::filesystem::path path;
        Mapping mapping;
        MaterialLayout layout;
        std::vector<AlignedBuffer<std::byte>> expanded;
        std::vector<unsigned char> gate_claimed, input_claimed, shuffle_claimed, mask_claimed,
            multiply_claimed, bitmul_claimed;
        std::vector<std::vector<unsigned char>> comparison_claimed;
        std::vector<unsigned char> evaluation_claimed;
        bool preloaded = false, execution_claimed = false, online = false;

        Impl(const MaterialPlan &p, unsigned id, const std::filesystem::path &file)
            : plan(p), party(id), mode(MaterialMode::Random), path(file), mapping(file, 0, false),
              layout(p, id, MaterialMode::Random) {
            const auto data = mapping.span();
            if (std::memcmp(data.data(), magic, 8) || get64(data, 8) != 2 ||
                get64(data, 16) != p.fingerprint() || get32(data, 24) != id ||
                get32(data, 28) != p.n) {
                throw std::runtime_error("material header/plan/party mismatch");
            }
            const auto raw_mode = get32(data, 32);
            if (raw_mode > 1) {
                throw std::runtime_error("invalid material mode");
            }
            mode = static_cast<MaterialMode>(raw_mode);
            layout = MaterialLayout(p, id, mode);
            if (get32(data, 36) != layout.segments.size() || get64(data, 40) != mapping.size ||
                mapping.size != layout.file_bytes || get64(data, 80) != layout.logical_share_bits ||
                get64(data, 88) != layout.gate_bits ||
                get64(data, 96) != layout.dense_share_bytes ||
                get64(data, 104) != layout.owner_bytes) {
                throw std::runtime_error("material layout size mismatch");
            }
            std::array<std::byte, record_bytes> expected{};
            for (const auto &s : layout.segments) {
                encode_record(s, expected);
                if (std::memcmp(expected.data(),
                                data.data() + header_bytes + s.id * record_bytes,
                                record_bytes)) {
                    throw std::runtime_error("material segment directory mismatch");
                }
            }
            {
                std::ifstream ready(file.parent_path() / "READY", std::ios::binary);
                std::array<std::byte, 24> marker{};
                if (!ready.read(reinterpret_cast<char *>(marker.data()), marker.size()) ||
                    ready.peek() != EOF || std::memcmp(marker.data(), data.data() + 48, 16) ||
                    get64(marker, 16) != p.fingerprint()) {
                    throw std::runtime_error("material package not complete/run marker mismatch");
                }
            }
            bitmul_claimed.resize(p.bitmuls.size(), 0);
            multiply_claimed.resize(p.multiplications.size(), 0);
            expanded.resize(layout.segments.size());
            gate_claimed.resize(p.gates.size(), 0);
            input_claimed.resize(p.inputs.size(), 0);
            shuffle_claimed.resize(p.shuffles.size(), 0);
            mask_claimed.resize(p.masks.size(), 0);
            for (const auto &c : p.comparisons) {
                comparison_claimed.emplace_back(c.evaluations, 0);
            }
            evaluation_claimed.resize(layout.segments.size(), 0);
        }
    };

    MaterialStore::MaterialStore(const MaterialPlan &p,
                                 unsigned id,
                                 const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(p, id, path)) {
    }

    MaterialStore::~MaterialStore() = default;

    const MaterialLayout &MaterialStore::layout() const noexcept {
        return impl_->layout;
    }

    MaterialMode MaterialStore::mode() const noexcept {
        return impl_->mode;
    }

    unsigned MaterialStore::party() const noexcept {
        return impl_->party;
    }

    std::span<const std::byte, 16> MaterialStore::run_id() const noexcept {
        return std::span<const std::byte, 16>(impl_->mapping.data + 48, 16);
    }

    void MaterialStore::preload() {
        if (impl_->online) {
            throw std::logic_error("material preparation after online start");
        }
        if (impl_->preloaded) {
            return;
        }
        Prg::Key key{};
        std::memcpy(key.data(), impl_->mapping.data + 64, 16);
        Prg prg(key);
        volatile unsigned touch = 0;
        for (const auto &s : impl_->layout.segments) {
            if (s.storage == SegmentStorage::Seeded) {
                auto &out = impl_->expanded[s.id];
                out = AlignedBuffer<std::byte>(s.bytes);
                prg.fill(s.id + 1, 0, out.span());
                trim(out.span(), s.bits);
            } else if (s.storage == SegmentStorage::Dense) {
                for (std::size_t i = 0; i < s.bytes; i += 4096) {
                    touch = touch ^ std::to_integer<unsigned>(impl_->mapping.data[s.offset + i]);
                }
            }
        }
        (void)touch;
        impl_->preloaded = true;
    }

    void MaterialStore::claim_comparison(std::size_t c, unsigned evaluation) {
        if (impl_->online || impl_->comparison_claimed.at(c).at(evaluation)) {
            throw std::logic_error("comparison masks already claimed or online");
        }
        const auto &segments = impl_->layout.comparisons.at(c).evaluations.at(evaluation);
        for (auto id : segments) {
            if (impl_->evaluation_claimed.at(id)) {
                throw std::logic_error("comparison physical evaluation already claimed");
            }
        }
        for (auto id : segments) {
            impl_->evaluation_claimed[id] = 1;
        }
        impl_->comparison_claimed[c][evaluation] = 1;
    }

    void MaterialStore::claim_bitmul(std::size_t index) {
        if (impl_->online || impl_->bitmul_claimed.at(index)) {
            throw std::logic_error("bitmul already claimed or online");
        }
        impl_->bitmul_claimed[index] = 1;
    }

    void MaterialStore::claim_multiply(std::size_t index) {
        if (impl_->online || impl_->multiply_claimed.at(index)) {
            throw std::logic_error("multiplication triple already claimed or online");
        }
        impl_->multiply_claimed[index] = 1;
    }

    void MaterialStore::claim_mask(std::size_t mask) {
        if (impl_->online || impl_->mask_claimed.at(mask)) {
            throw std::logic_error("fresh positional mask already claimed or online");
        }
        impl_->mask_claimed[mask] = 1;
    }

    void MaterialStore::claim_shuffle(std::size_t shuffle) {
        if (impl_->online || impl_->shuffle_claimed.at(shuffle)) {
            throw std::logic_error("shuffle material already claimed or online");
        }
        impl_->shuffle_claimed[shuffle] = 1;
    }

    void MaterialStore::claim_input(std::size_t input) {
        if (impl_->online || impl_->input_claimed.at(input)) {
            throw std::logic_error("input mask already claimed or online");
        }
        impl_->input_claimed[input] = 1;
    }

    void MaterialStore::claim_gate(std::size_t gate) {
        if (impl_->online || impl_->gate_claimed.at(gate)) {
            throw std::logic_error("correlated gate material already claimed or online");
        }
        impl_->gate_claimed[gate] = 1;
    }

    void MaterialStore::claim_execution() {
        if (impl_->execution_claimed || impl_->online) {
            throw std::logic_error("execution already claimed");
        }
        const auto lock = impl_->path.string() + ".used";
        int fd = ::open(lock.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) {
            throw std::runtime_error(
                "one-time material already used or cannot create execution lease");
        }
        const auto n = ::write(fd, run_id().data(), 16);
        const auto synced = durable_material() ? ::fsync(fd) : 0;
        const auto closed = ::close(fd);
        const auto parent_path =
            impl_->path.has_parent_path() ? impl_->path.parent_path() : std::filesystem::path(".");
        const int parent = ::open(parent_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        const bool directory_error = parent < 0 || (durable_material() && ::fsync(parent) != 0);
        if (parent >= 0) {
            ::close(parent);
        }
        if (n != 16 || synced || closed || directory_error) {
            throw std::runtime_error("cannot persist one-time material lease");
        }
        impl_->execution_claimed = true;
    }

    void MaterialStore::begin_online() {
        if (const auto *path = std::getenv("BJ_X11_MASK_AUDIT")) {
            nlohmann::json trace = nlohmann::json::array();
            std::set<std::string> ids;
            auto add = [&](std::string id, std::size_t count) {
                if (!ids.insert(id).second) {
                    throw std::logic_error("duplicate mask identifier");
                }
                trace.push_back({{"id", id}, {"count", count}});
            };
            for (std::size_t i = 0; i < impl_->input_claimed.size(); ++i) {
                if (impl_->input_claimed[i]) {
                    add("input/" + std::to_string(i),
                        impl_->layout.segments[impl_->layout.input_shares[i]].count);
                }
            }
            for (std::size_t i = 0; i < impl_->comparison_claimed.size(); ++i) {
                for (std::size_t e = 0; e < impl_->comparison_claimed[i].size(); ++e) {
                    if (impl_->comparison_claimed[i][e]) {
                        for (auto s : impl_->layout.comparisons[i].evaluations[e]) {
                            add("comparison/evaluation/segment/" + std::to_string(s),
                                impl_->layout.segments[s].count);
                        }
                        if (e == 0) {
                            for (auto s : impl_->layout.comparisons[i].aux) {
                                add("comparison/aux/segment/" + std::to_string(s),
                                    impl_->layout.segments[s].count);
                            }
                        }
                    }
                }
            }
            for (std::size_t i = 0; i < impl_->gate_claimed.size(); ++i) {
                if (impl_->gate_claimed[i]) {
                    auto s = impl_->layout.gate_masks[i];
                    add("gate/segment/" + std::to_string(s), impl_->layout.segments[s].count);
                }
            }
            for (std::size_t i = 0; i < impl_->mask_claimed.size(); ++i) {
                if (impl_->mask_claimed[i]) {
                    auto s = impl_->layout.masks[i];
                    add("position/segment/" + std::to_string(s), impl_->layout.segments[s].count);
                }
            }
            std::ofstream f(path);
            f << nlohmann::
                     json{{"mask_ranges", trace},
                          {"unique", true},
                          {"guard",
                           "MaterialStore physical claim guards plus one-time contribute guards"}}
                         .dump()
              << '\n';
        }

        if (impl_->online || !impl_->preloaded || !impl_->execution_claimed) {
            throw std::logic_error("offline preparation/one-time lease required");
        }
        impl_->online = true;
    }

    std::span<const std::byte> MaterialStore::packed(std::size_t id) const {
        if (impl_->online || !impl_->preloaded) {
            throw std::logic_error("material access outside offline preparation");
        }
        const auto &s = impl_->layout.segments.at(id);
        if (s.storage == SegmentStorage::Absent) {
            throw std::logic_error("private owner mask is not assigned to this party");
        }
        if (s.storage == SegmentStorage::Seeded) {
            if (impl_->expanded[id].size() != s.bytes) {
                throw std::logic_error("X11 packed segment accessed after decoded replica release");
            }
            return impl_->expanded[id].span();
        }
        return impl_->mapping.span().subspan(s.offset, s.bytes);
    }
}

namespace bj {
    void MaterialStore::release_decoded_segments(std::span<const std::size_t> ids) {
        if (impl_->online || !impl_->preloaded) {
            throw std::logic_error("X11 release outside preparation");
        }
        for (auto id : ids) {
            const auto &s = impl_->layout.segments.at(id);
            if (s.storage == SegmentStorage::Seeded) {
                impl_->expanded[id] = AlignedBuffer<std::byte>();
            }
        }
    }
}
