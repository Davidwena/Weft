#include "bj/common/timer.hpp"
#include "bj/data/bank.hpp"
#include "bj/ref/investigate.hpp"
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/resource.h>
using Json = nlohmann::json;

int main(int argc, char **argv) {
    try {
        std::string data, query, subject_file, inputs;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) {
                throw std::invalid_argument("reference options require values");
            }
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "--data") {
                data = v;
            } else if (k == "--query") {
                query = v;
            } else if (k == "--subject-file") {
                subject_file = v;
            } else if (k == "--inputs") {
                inputs = v;
            } else {
                throw std::invalid_argument("unknown reference option");
            }
        }
        const auto p = bj::Plan::load(query);
        std::ifstream sf(subject_file);
        if (!sf) {
            throw std::runtime_error("missing private subject");
        }
        const auto subject = Json::parse(sf).at("subject").get<std::uint64_t>();
        bj::Timer timer;
        std::vector<bj::ref::FilteredRow> rows;

        for (unsigned bank = 0; bank < p.common.n; ++bank) {
            bj::BankData file(std::filesystem::path(data) /
                              ("bank_" + std::to_string(bank) + ".bjcol"));
            if (file.bank() != bank || file.parties() != p.common.n ||
                file.size() != p.common.rows[bank]) {
                throw std::invalid_argument("reference bank/config mismatch");
            }
            Json flags = Json::object();
            if (!p.common.filters.empty()) {
                std::ifstream f(std::filesystem::path(inputs) /
                                ("input_" + std::to_string(bank) + ".json"));
                if (!f) {
                    throw std::runtime_error("reference filters require --inputs");
                }
                flags = Json::parse(f).at("inputs");
                for (const auto &name : p.common.filters) {
                    const auto key = "f_" + name + "_" + std::to_string(bank);
                    if (!flags.contains(key) || !flags[key].is_array() ||
                        flags[key].size() != file.size()) {
                        throw std::invalid_argument("reference filter shape");
                    }
                }
            }
            for (std::size_t i = 0; i < file.size(); ++i) {
                if (file.sid(i) != subject) {
                    continue;
                }
                bj::ref::FilteredRow row;
                row.row = {file.sid(i),
                           file.time(i),
                           file.amount(i),
                           file.outgoing(i),
                           static_cast<std::uint64_t>(file.counterparty(i))};
                for (std::size_t f = 0; f < p.common.filters.size(); ++f) {
                    const auto value =
                        flags["f_" + p.common.filters[f] + "_" + std::to_string(bank)][i];
                    bool bit = false;
                    if (value.is_boolean()) {
                        bit = value.get<bool>();
                    } else if (value.is_number_unsigned() && value.get<std::uint64_t>() <= 1) {
                        bit = value.get<std::uint64_t>() != 0;
                    } else if (value.is_number_integer() && value.get<std::int64_t>() >= 0 &&
                               value.get<std::int64_t>() <= 1) {
                        bit = value.get<std::int64_t>() != 0;
                    } else {
                        throw std::invalid_argument("reference filter must be Boolean");
                    }
                    row.filters |= static_cast<std::uint16_t>(static_cast<unsigned>(bit) << f);
                }
                rows.push_back(row);
            }
        }
        const auto r = bj::ref::evaluate(p, rows, subject);
        Json output = {
            {"omega", r.overflow},
            {"full_in", r.full_in},
            {"full_out", r.full_out},
            {"retained_in", r.retained_in},
            {"retained_out", r.retained_out},
            {"templates", Json::object()},
            {"expected",
             {{"omega", static_cast<unsigned>(r.overflow)}, {"templates", Json::object()}}}};
        for (std::size_t i = 0; i < r.templates.size(); ++i) {
            const auto &t = r.templates[i];
            output["templates"][t.name] = {{"full", bj::ref::decimal(t.full)},
                                           {"retained", bj::ref::decimal(t.retained)},
                                           {"full_alarm", t.full_alarm},
                                           {"retained_alarm", t.retained_alarm},
                                           {"lower_bound", true}};
            output["expected"]["templates"][t.name] =
                p.templates[i].config.output == "alarm"
                    ? static_cast<std::uint64_t>(t.retained_alarm)
                    : static_cast<std::uint64_t>(t.retained);
        }
        rusage usage{};
        ::getrusage(RUSAGE_SELF, &usage);
        output["seconds"] = timer.seconds();
        output["peak_rss_kib"] = usage.ru_maxrss;
        std::puts(output.dump().c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "plain_reference: %s\n", e.what());
        return 1;
    }
}
