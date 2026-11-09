#include "bj/plan/investigation.hpp"
#include "bj/common/config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <numeric>
#include <bit>
#include <cmath>

namespace bj {
    using Json = nlohmann::json;

    std::size_t InvestigationConfig::size() const {
        std::size_t n = 0;
        for (auto v : rows) {
            if (v > SIZE_MAX - n) {
                throw std::invalid_argument("row count overflow");
            }
            n += v;
        }
        return n;
    }

    std::size_t InvestigationConfig::pairs() const {
        if (y_size() && x_size() > SIZE_MAX / y_size()) {
            throw std::invalid_argument("pair count overflow");
        }
        return x_size() * y_size();
    }

    InvestigationConfig InvestigationConfig::parse(const Json &document) {
        const auto &j = document.at("investigation");
        InvestigationConfig c;
        c.n = j.value("n", 3u);
        if (c.n < 2 || c.n > 1024) {
            throw std::invalid_argument("bank count outside supported range");
        }
        c.king = j.value("king", 0u);
        c.initiator = j.value("initiator", c.n - 1);
        c.rows = j.value("rows", std::vector<std::size_t>(c.n, 0));
        c.kin = j.value("kin", 512ul);
        c.kout = j.value("kout", 512ul);
        c.bt = j.value("bt", 29u);
        c.ba = j.value("ba", 40u);
        c.bq = j.value("bq", 7u);
        c.kt = j.value("kt", 32u);
        c.ka = j.value("ka", 49u);
        if (c.size() > (std::size_t{1} << 28) || c.kin > (1u << 20) || c.kout > (1u << 20)) {
            throw std::invalid_argument("public investigation shape exceeds supported range");
        }
        const auto max_prefix = c.size() + std::max(c.kin, c.kout);
        c.kp = j.value("kp", std::max(22u, static_cast<unsigned>(std::bit_width(max_prefix)) + 1));
        c.p3_query_begin = j.value("p3_query_begin", std::size_t{0});
        c.p3_query_count = j.value("p3_query_count", SIZE_MAX);
        if (j.contains("p3_query_count") &&
            (!j["p3_query_count"].is_number_integer() || c.p3_query_count == SIZE_MAX)) {
            throw std::invalid_argument("P3 batch count must be a positive public integer");
        }
        c.p = j.value("p", 5ull);
        c.q = j.value("q", 100ull);
        c.threshold = j.value("threshold", 1ull);
        c.mode = j.value("mode", std::string("pipeline"));
        c.variant = j.value("variant", std::string("C1+C2"));
        c.open_mode = j.value("open_mode", std::string("auto"));
        c.output = j.value("output", std::string("count"));
        if (j.contains("windows")) {
            c.windows.clear();
            for (const auto &w : j.at("windows")) {
                c.windows.push_back({w.at(0), w.at(1)});
            }
        }
        c.pattern = j.value("pattern", std::string("P1"));
        c.amount_lower = j.value("a_lo", 1ull);
        c.amount_upper = j.value("a_hi", c.ba < 64 ? std::uint64_t{1} << c.ba : 0ull);
        c.min_cluster = j.value("m", 3ull);
        c.algorithm = j.value("operator", std::string("nl"));
        c.dc_pool_factor = j.value("dc_pool_factor", 1.5);
        c.dc_pool_safe = j.value("dc_pool_safe", false);
        c.filters = j.value("carried_filters", std::vector<std::string>{});
        c.validate();
        return c;
    }

    InvestigationConfig InvestigationConfig::load(const std::filesystem::path &path) {
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("cannot read investigation plan");
        }
        Json j = Json::parse(f);
        if (j.contains("context")) {
            j = Json::parse(j.at("context").get<std::string>());
        }
        return parse(j);
    }

    Json InvestigationConfig::json() const {
        Json w = Json::array();
        for (auto x : windows) {
            w.push_back({x.lower, x.upper});
        }
        Json result = {{"investigation",
                        {{"n", n},
                         {"king", king},
                         {"initiator", initiator},
                         {"rows", rows},
                         {"kin", kin},
                         {"kout", kout},
                         {"bt", bt},
                         {"ba", ba},
                         {"bq", bq},
                         {"kt", kt},
                         {"ka", ka},
                         {"kp", kp},
                         {"p", p},
                         {"q", q},
                         {"threshold", threshold},
                         {"mode", mode},
                         {"variant", variant},
                         {"open_mode", open_mode},
                         {"output", output},
                         {"windows", w}}}};
        if (algorithm == "dc") {
            auto &j = result["investigation"];
            j["operator"] = algorithm;
            j["dc_pool_factor"] = dc_pool_factor;
            j["dc_pool_safe"] = dc_pool_safe;
        }
        if (pattern != "P1") {
            auto &j = result["investigation"];
            j["pattern"] = pattern;
            j["a_lo"] = amount_lower;
            j["a_hi"] = amount_upper;
            j["m"] = min_cluster;
        }
        if (!filters.empty()) {
            result["investigation"]["carried_filters"] = filters;
        }
        if (p3_query_count != SIZE_MAX) {
            result["investigation"]["p3_query_begin"] = p3_query_begin;
            result["investigation"]["p3_query_count"] = p3_query_count;
        }
        return result;
    }

    void InvestigationConfig::validate() const {
        if (p3_query_count != SIZE_MAX &&
            (pattern != "P3" || mode != "band" || algorithm != "nl" || !filters.empty() ||
             p3_query_begin > kin || !p3_query_count || p3_query_count > kin - p3_query_begin)) {
            throw std::invalid_argument("invalid public P3 NL query batch");
        }
        if (p3_query_count == SIZE_MAX && p3_query_begin) {
            throw std::invalid_argument("P3 batch offset without count");
        }

        {
            std::vector<std::string> seen;
            for (const auto &f : filters) {
                if (f.empty() || f.size() > 32 ||
                    !std::all_of(f.begin(),
                                 f.end(),
                                 [](unsigned char ch) {
                                     return std::isalnum(ch) || ch == '_';
                                 }) ||
                    std::find(seen.begin(), seen.end(), f) != seen.end()) {
                    throw std::invalid_argument("invalid carried filter name");
                }
                seen.push_back(f);
            }
            if (filters.size() > 16 || (!filters.empty() && mode == "band")) {
                throw std::invalid_argument("carried filters require pipeline/compact");
            }
        }
        if (n < 2 || n > 1024 || rows.size() != n || king >= n || initiator >= n || bt < 1 ||
            bt > 61 || ba < 1 || ba > 62 || bq < 1 || bq > 30 || kt < 2 || kt > 64 || ka < 2 ||
            ka > (algorithm == "dc" ? 72u : 64u) || kp < 2 || kp > 32) {
            throw std::invalid_argument("invalid investigation dimensions");
        }
        if (windows.empty() || windows.size() > 64) {
            throw std::invalid_argument("investigation requires 1..64 public windows");
        }
        if (pattern != "P1" && pattern != "P2" && pattern != "P3" && pattern != "P4") {
            throw std::invalid_argument("unknown query pattern");
        }
        if (pattern != "P1" && (windows.size() != 1 || variant != "C1+C2")) {
            throw std::invalid_argument("pattern templates require one window and C1+C2");
        }
        if (pattern == "P3" &&
            (amount_lower >= amount_upper || amount_upper > (std::uint64_t{1} << ba) ||
             min_cluster < 2 || min_cluster > (1ull << 31) || windows[0].lower != 0)) {
            throw std::invalid_argument("P3 amount/degree/window range");
        }
        if (pattern == "P2" && uint128_t(kout) * (uint128_t{1} << ba) >= (uint128_t{1} << 63)) {
            throw std::invalid_argument("P2 amount sum range");
        }
        if (algorithm != "nl" && algorithm != "dc") {
            throw std::invalid_argument("unknown join operator");
        }
        if (!q || p >= q || uint128_t(q) + p >= 256 || q >= (std::uint64_t{1} << bq)) {
            throw std::invalid_argument("invalid public tolerance range");
        }

        if (kin > (1u << 20) || kout > (1u << 20)) {
            throw std::invalid_argument("slot bound exceeds 2^20");
        }
        if (algorithm == "dc") {
            const auto elements = x_size() + 2 * y_size();
            if (pattern != "P4" && elements > (std::size_t{1} << 21)) {
                throw std::invalid_argument("DC M exceeds paper bound 2^21");
            }
            const auto h = static_cast<unsigned>(std::bit_width(elements ? elements - 1 : 0));
            if (windows.size() != 1 || !std::isfinite(dc_pool_factor) || dc_pool_factor <= 0 ||
                dc_pool_factor > 100) {
                throw std::invalid_argument(
                    "DC requires one window and a finite public budget factor");
            }
            const uint128_t id = elements ? elements - 1 : 0, scale = uint128_t{1} << h;
            uint128_t time_max = 3 * (2 * (uint128_t{1} << bt) + windows[0].upper + 1) + 2;
            uint128_t amount_max = 3 * (uint128_t(q) + p) * ((uint128_t{1} << ba) - 1) + 2;
            if (pattern == "P4") {

                const uint128_t groups = x_size() + y_size();
                const auto group_max = groups ? groups - 1 : 0;
                time_max += group_max * (uint128_t{1} << (bt + 4));
                const auto cp_max = (uint128_t{UINT64_MAX} << (h + 1)) + scale + id;
                unsigned cp_bits = 1;
                for (auto v = cp_max; v; v >>= 1) {
                    ++cp_bits;
                }
                if (cp_bits > 96) {
                    throw std::invalid_argument("DC counterparty key needs " +
                                                std::to_string(cp_bits) +
                                                " signed bits; comparison ring has 96 bits");
                }
            }
            auto check_key = [&](uint128_t coordinate, unsigned width, const char *name) {
                const auto maximum = coordinate * scale + id;
                unsigned needed = 1;
                for (auto v = maximum; v; v >>= 1) {
                    ++needed;
                }
                if (needed > width) {
                    throw std::invalid_argument(
                        std::string("DC ") + name + " key needs " + std::to_string(needed) +
                        " signed bits; comparison ring has " + std::to_string(width) + " bits");
                }
            };
            check_key(time_max, 64, "time");
            if (pattern != "P3") {
                check_key(amount_max, 72, "amount");
            }
        }
        parse_open_mode(open_mode);
        if (mode != "band" && mode != "compact" && mode != "pipeline") {
            throw std::invalid_argument("investigation mode");
        }
        if (variant != "C1" && variant != "C1+zero" && variant != "+interval" &&
            variant != "+amount" && variant != "C1+C2") {
            throw std::invalid_argument("unknown band variant");
        }
        if (output != "count" && output != "alarm") {
            throw std::invalid_argument("output policy");
        }
        if (!q || p >= q || q + p >= 256 || q >= (std::uint64_t{1} << bq) ||
            (algorithm == "nl" &&
             uint128_t(q + p) * ((uint128_t{1} << ba) - 1) >= (uint128_t{1} << (ka - 1))) ||
            windows.empty() || windows.size() > 64 || (variant != "C1+C2" && windows.size() != 1)) {
            throw std::invalid_argument("band amount/window parameters");
        }
        if (pairs() >= (std::uint64_t{1} << (algorithm == "dc" && output == "count" ? 63 : 31)) ||
            threshold > (std::uint64_t{1} << (pattern == "P2" ? 63 : 31)) ||
            size() > (std::size_t{1} << 28) || kin > (1u << 20) || kout > (1u << 20)) {
            throw std::invalid_argument("count/shape exceeds signed range");
        }
        if (mode != "band" &&
            (!(size() + kin + kout) ||
             uint128_t(size()) + std::max(kin, kout) + 1 > (uint128_t{1} << (kp - 1)))) {
            throw std::invalid_argument("prefix ring too narrow");
        }
        for (auto w : windows) {
            if (w.lower > w.upper || w.upper >= (std::uint64_t{1} << bt) ||
                (algorithm == "nl" &&
                 uint128_t(2) * (uint128_t{1} << bt) + w.upper + 1 > (uint128_t{1} << (kt - 1)))) {
                throw std::invalid_argument("time interval/range condition violated");
            }
        }
    }

    std::size_t named_mask(const MaterialPlan &p, const std::string &name) {
        for (std::size_t i = 0; i < p.inputs.size(); ++i) {
            if (p.inputs[i].name == name) {
                return i;
            }
        }
        for (std::size_t i = 0; i < p.masks.size(); ++i) {
            if (p.masks[i].name == name) {
                return p.inputs.size() + i;
            }
        }
        throw std::invalid_argument("unknown mask " + name);
    }

    std::size_t named_comparison(const MaterialPlan &p, const std::string &name) {
        for (std::size_t i = 0; i < p.comparisons.size(); ++i) {
            if (p.comparisons[i].name == name) {
                return i;
            }
        }
        throw std::invalid_argument("unknown comparison " + name);
    }

    std::size_t named_gate(const MaterialPlan &p, const std::string &name) {
        for (std::size_t i = 0; i < p.gates.size(); ++i) {
            if (p.gates[i].name == name) {
                return i;
            }
        }
        throw std::invalid_argument("unknown gate " + name);
    }

    std::vector<std::uint64_t> window_offsets(const InvestigationConfig &c) {
        std::vector<std::uint64_t> offsets;
        for (auto w : c.windows) {
            offsets.push_back(w.lower);
            offsets.push_back(w.upper + 1);
        }
        std::sort(offsets.begin(), offsets.end());
        offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
        return offsets;
    }

    MaterialPlan compile_investigation(const InvestigationConfig &c) {
        c.validate();
        MaterialPlan p;
        p.n = c.n;
        p.king = c.king;
        p.output_party = c.initiator;
        p.open_mode = c.open_mode;
        p.context = c.json().dump();
        auto input = [&](std::string name, unsigned owner, unsigned width, std::size_t count) {
            p.inputs.push_back({std::move(name), owner, width, count});
        };
        if (c.mode == "band") {
            input("tin", 0, 64, c.kin);
            input("ain", 0, 96, c.kin);
            input("tout", c.n - 1, 64, c.kout);
            input("aout", c.n - 1, 96, c.kout);
            input("omega_in", 0, 0, 1);
            input("omega_out", c.n - 1, 0, 1);
        } else {
            for (unsigned bank = 0; bank < c.n; ++bank) {
                const auto suffix = "_" + std::to_string(bank);
                input("sid" + suffix, bank, 64, c.rows[bank]);
                input("dir" + suffix, bank, 0, c.rows[bank]);
                input("t" + suffix, bank, 64, c.rows[bank]);
                input("a" + suffix, bank, 96, c.rows[bank]);
            }
            input("subject", c.initiator, 64, 1);
        }
        if (c.carry_cp()) {
            if (c.mode == "band") {
                input("cpin", 0, 96, c.kin);
                input("cpout", c.n - 1, 96, c.kout);
            } else {
                for (unsigned bank = 0; bank < c.n; ++bank) {
                    input("cp_" + std::to_string(bank), bank, 96, c.rows[bank]);
                }
            }
        }
        if ((c.algorithm == "dc" || c.pattern != "P1") && c.mode == "band") {
            input("vin", 0, 0, c.kin);
            input("vout", c.n - 1, 0, c.kout);
        }
        for (const auto &f : c.filters) {
            for (unsigned bank = 0; bank < c.n; ++bank) {
                input("f_" + f + "_" + std::to_string(bank), bank, 0, c.rows[bank]);
            }
        }
        auto mask = [&](std::string name,
                        unsigned width,
                        std::size_t count,
                        bool fresh = true,
                        std::vector<MaskTerm> terms = {}) {
            p.masks.push_back({std::move(name), width, count, fresh, fresh, std::move(terms)});
            return p.inputs.size() + p.masks.size() - 1;
        };
        auto compare = [&](std::string name,
                           CompareKind kind,
                           unsigned width,
                           std::size_t count,
                           std::vector<std::size_t> refs = {},
                           unsigned eval = 1) {
            CompareSpec s;
            s.name = std::move(name);
            s.kind = kind;
            s.width = width;
            s.count = count;
            s.inputs = std::move(refs);
            s.evaluations = eval;
            p.comparisons.push_back(std::move(s));
        };
        auto gate = [&](std::string name, unsigned m, unsigned width, std::size_t count) {
            p.gates.push_back({std::move(name), m, width, count, {}});
        };
        if (c.mode != "band") {
            const auto N = c.size(), M = N + c.kin + c.kout;
            std::vector<MaskTerm> sid, time, amount, counterparty;
            std::size_t offset = 0;
            for (unsigned bank = 0; bank < c.n; ++bank) {
                const auto suffix = "_" + std::to_string(bank);
                const auto length = c.rows[bank];
                if (length) {
                    sid.push_back({named_mask(p, "sid" + suffix), 0, offset, length, 1, length, 1});
                    time.push_back({named_mask(p, "t" + suffix), 0, offset, length, 1, length, 1});
                    amount.push_back(
                        {named_mask(p, "a" + suffix), 0, offset, length, 1, length, 1});
                    if (c.carry_cp()) {
                        counterparty.push_back(
                            {named_mask(p, "cp" + suffix), 0, offset, length, 1, length, 1});
                    }
                }
                offset += length;
            }
            sid.push_back({named_mask(p, "subject"), 0, 0, N, 1, 1, -1});
            auto eq = mask("sid_difference", 64, N, false, sid);
            compare("compact.eq", CompareKind::Eq, 64, N, {eq});
            for (auto label : {"in", "out"}) {
                std::string name = label;
                auto K = name == "in" ? c.kin : c.kout;
                gate("compact.select." + name, c.filters.empty() ? 2 : 3, 32, N);
                compare("compact.prefix." + name, CompareKind::MsbFresh, c.kp, N + 1);
                gate("compact.keep." + name, c.filters.empty() ? 3 : 4, 32, N);
                compare("compact.dummy." + name, CompareKind::MsbFresh, c.kp, K);
                gate("compact.dummy_b2a." + name, 1, 32, K);
            }
            auto t = mask("table.time", 64, M, false, time),
                 a = mask("table.amount", 96, M, false, amount), v = mask("table.vld", 0, M, false);
            auto fi = mask("table.fin", 32, M), fo = mask("table.fout", 32, M);
            std::vector<std::size_t> outputs;
            for (auto [name, width] : std::vector<std::pair<std::string, unsigned>>{
                     {"time", 64}, {"amount", 96}, {"vld", 0}, {"fin", 32}, {"fout", 32}}) {
                outputs.push_back(mask("shuffle." + name, width, M));
            }
            std::vector<std::size_t> columns{t, a, v, fi, fo};
            if (c.carry_cp()) {
                columns.push_back(mask("table.cp", 96, M, false, counterparty));
                outputs.push_back(mask("shuffle.cp", 96, M));
            }
            for (const auto &f : c.filters) {
                std::vector<MaskTerm> terms;
                std::size_t off = 0;
                for (unsigned bank = 0; bank < c.n; ++bank) {
                    const auto len = c.rows[bank];
                    if (len) {
                        terms.push_back({named_mask(p, "f_" + f + "_" + std::to_string(bank)),
                                         0,
                                         off,
                                         len,
                                         1,
                                         len,
                                         1});
                    }
                    off += len;
                }
                columns.push_back(mask("table.f_" + f, 0, M, false, terms));
                outputs.push_back(mask("shuffle.f_" + f, 0, M));
            }
            p.shuffles.push_back({"compact", M, columns, outputs});
            for (auto label : {"in", "out"}) {
                std::string suffix = label;
                const auto K = suffix == "in" ? c.kin : c.kout;
                mask("slot.t" + suffix, 64, K);
                mask("slot.a" + suffix, 96, K);
                mask("slot.v" + suffix, 0, K);
                if (c.carry_cp()) {
                    mask("slot.cp" + suffix, 96, K);
                }
                for (const auto &f : c.filters) {
                    mask("slot.f_" + f + "_" + suffix, 0, K);
                }
            }
        }
        if (c.mode != "compact" && c.pattern != "P1") {
            compile_pattern_material(c, p);
        }
        if (c.mode != "compact" && c.algorithm == "dc" && c.pattern == "P1") {
            compile_dc_material(c, p);
        }
        if (c.mode != "compact" && c.algorithm == "nl" && c.pattern == "P1") {
            const auto P = c.pairs();
            auto tin = named_mask(p, c.mode == "band" ? "tin" : "slot.tin"),
                 tout = named_mask(p, c.mode == "band" ? "tout" : "slot.tout"),
                 ain = named_mask(p, c.mode == "band" ? "ain" : "slot.ain"),
                 aout = named_mask(p, c.mode == "band" ? "aout" : "slot.aout");
            const auto stride = std::max(std::size_t{1}, c.kout);
            auto terms = [&](std::size_t out, std::int64_t co, std::size_t in, std::int64_t ci) {
                return std::vector<MaskTerm>{
                    {out, 0, 0, P, 1, stride, co},
                    {in, 0, 0, P, stride, std::max(std::size_t{1}, c.kin), ci}};
            };
            auto rt = mask("pair.time", c.kt, P, false, terms(tout, 1, tin, -1));
            auto ra = mask("pair.amount1",
                           c.ka,
                           P,
                           false,
                           terms(aout,
                                 -static_cast<std::int64_t>(c.q),
                                 ain,
                                 static_cast<std::int64_t>(c.q + c.p)));
            auto rb = mask("pair.amount2",
                           c.ka,
                           P,
                           false,
                           terms(aout,
                                 static_cast<std::int64_t>(c.q),
                                 ain,
                                 -static_cast<std::int64_t>(c.q - c.p)));
            if (c.interval()) {
                compare("band.interval",
                        c.zero_open() ? CompareKind::Interval : CompareKind::IntervalFresh,
                        c.kt,
                        P,
                        c.zero_open() ? std::vector<std::size_t>{rt} : std::vector<std::size_t>{},
                        static_cast<unsigned>(window_offsets(c).size()));
            } else {
                for (unsigned j = 0; j < 2; ++j) {
                    compare("band.time" + std::to_string(j),
                            c.zero_open() ? CompareKind::MsbKnown : CompareKind::MsbFresh,
                            c.kt,
                            P,
                            c.zero_open() ? std::vector<std::size_t>{rt}
                                          : std::vector<std::size_t>{});
                }
            }
            compare("band.amount1",
                    c.zero_open() ? CompareKind::MsbKnown : CompareKind::MsbFresh,
                    c.ka,
                    P,
                    c.zero_open() ? std::vector<std::size_t>{ra} : std::vector<std::size_t>{});
            compare("band.amount2",
                    c.zero_open() ? CompareKind::MsbKnown : CompareKind::MsbFresh,
                    c.ka,
                    P,
                    c.zero_open() ? std::vector<std::size_t>{rb} : std::vector<std::size_t>{});
            const auto arity = (c.interval() ? 1u : 2u) + (c.amount() ? 1u : 2u);
            for (std::size_t w = 0; w < c.windows.size(); ++w) {
                gate("band.product." + std::to_string(w), arity, 64, P);
            }
            if (c.output == "alarm") {
                compare("band.alarm", CompareKind::MsbFresh, 32, c.windows.size());
            }
        }
        gate("output.omega", 2, 64, 1);
        p.validate();
        return p;
    }
}
