#include "bj/ops/compact.hpp"
#include "bj/ops/nl.hpp"
#include "bj/ops/dc.hpp"
#include "bj/common/config.hpp"
#include "../tests/allocation.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <sys/resource.h>
#include <unistd.h>
#include <time.h>
#ifdef BJ_GPROF_DC
extern "C" void moncontrol(int);
#endif
namespace {
    using namespace bj;
    using Json = nlohmann::json;

    struct Snapshot {
        double seconds = 0;
        IoTiming io;
        OpenStats open;
        std::vector<PeerStats> peers;

        explicit Snapshot(unsigned n) : peers(n) {
        }

        void capture(const Net &net, double s) {
            seconds = s;
            io = net.io_timing();
            open = net.open_stats();
            std::copy(net.stats().begin(), net.stats().end(), peers.begin());
        }

        Json json() const {
            Json rows = Json::array();
            for (unsigned i = 0; i < peers.size(); ++i) {
                const auto &p = peers[i];
                rows.push_back({{"peer", i},
                                {"sent_payload", p.sent_payload},
                                {"received_payload", p.received_payload},
                                {"sent_wire", p.sent_wire},
                                {"received_wire", p.received_wire},
                                {"sent_messages", p.sent_messages},
                                {"received_messages", p.received_messages}});
            }
            return {{"seconds", seconds},
                    {"poll_wait_seconds", io.poll_wait_seconds},
                    {"socket_io_seconds", io.socket_io_seconds},
                    {"caller_blocked_seconds", io.caller_blocked_seconds},
                    {"local_compute_seconds",
                     std::max(0.0, seconds - io.poll_wait_seconds - io.socket_io_seconds)},
                    {"open_rounds", open.rounds},
                    {"network_hops", open.hops},
                    {"contribution_bits", open.contribution_bits},
                    {"peers", rows}};
        }
    };

    std::uint64_t
    comparison_bits(const MaterialPlan &p, const MaterialLayout &l, std::string prefix) {
        std::uint64_t result = 0;
        for (std::size_t i = 0; i < p.comparisons.size(); ++i) {
            if (p.comparisons[i].name.starts_with(prefix)) {
                const auto &ids = l.comparisons[i];
                for (auto x : ids.tables) {
                    result += l.segments[x].bits;
                }
                for (auto x : ids.aux) {
                    result += l.segments[x].bits;
                }
                for (const auto &row : ids.evaluations) {
                    for (auto x : row) {
                        result += l.segments[x].bits;
                    }
                }
            }
        }
        for (std::size_t i = 0; i < p.gates.size(); ++i) {
            if (p.gates[i].name.starts_with(prefix)) {
                result += l.segments[l.gate_masks[i]].bits;
                for (auto x : l.gate_products[i]) {
                    result += l.segments[x].bits;
                }
            }
        }
        return result;
    }

    void validate_owned(const InvestigationConfig &c,
                        const MaterialPlan &p,
                        const InputManager &in,
                        unsigned id) {
        for (std::size_t j = 0; j < p.inputs.size(); ++j) {
            const auto &spec = p.inputs[j];
            if (spec.owner != id) {
                continue;
            }
            const auto &data = in.at(j);
            if (spec.name.starts_with("a")) {
                for (auto value : data.value128.span()) {
                    if (value >= (uint128_t{1} << c.ba)) {
                        throw std::invalid_argument("owned amount outside public range");
                    }
                }
            }
            if (spec.name.starts_with("t")) {
                for (auto value : data.value64.span()) {
                    if (value >= (std::uint64_t{1} << c.bt) &&
                        !(c.mode == "band" && (value == (std::uint64_t{1} << c.bt) ||
                                               value == 0 - (std::uint64_t{1} << c.bt)))) {
                        throw std::invalid_argument("owned timestamp outside public range");
                    }
                }
            }
        }
    }

    void copy_slot(const PreparedInput &t, const PreparedInput &a, MaskedTable &table) {
        std::copy(
            t.delta64.span().begin(), t.delta64.span().end(), table.columns[0].d64.span().begin());
        std::copy(
            t.mask64.span().begin(), t.mask64.span().end(), table.columns[0].m64.span().begin());
        std::copy(a.delta128.span().begin(),
                  a.delta128.span().end(),
                  table.columns[1].d128.span().begin());
        std::copy(
            a.mask128.span().begin(), a.mask128.span().end(), table.columns[1].m128.span().begin());
    }
}

int main(int argc, char **argv) {
    try {
        using namespace bj;
        using Json = nlohmann::json;
#ifdef BJ_GPROF_DC
        moncontrol(0);
#endif
        std::string plan_file, material_file, input_file, hosts, bank_file;
        unsigned id = UINT32_MAX, n = 0, local_repeats = 0;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) {
                throw std::invalid_argument("options require values");
            }
            std::string key = argv[i], value = argv[i + 1];
            if (key == "--plan") {
                plan_file = value;
            } else if (key == "--material") {
                material_file = value;
            } else if (key == "--input") {
                input_file = value;
            } else if (key == "--hosts") {
                hosts = value;
            } else if (key == "--bank-file") {
                bank_file = value;
            } else if (key == "--id") {
                id = std::stoul(value);
            } else if (key == "--n") {
                n = std::stoul(value);
            } else if (key == "--local-repeats") {
                local_repeats = std::stoul(value);
            } else {
                throw std::invalid_argument("unknown investigation option");
            }
        }
        Timer planning;
        const auto c = InvestigationConfig::load(plan_file);
        const auto plan = MaterialPlan::load(plan_file);
        if (n != c.n || id >= n) {
            throw std::invalid_argument("bank count/id mismatch");
        }
        auto config = Config::load(hosts);
        config.king = c.king;
        config.open_mode = parse_open_mode(c.open_mode);
        config.validate(n);
        omp_set_dynamic(0);
        omp_set_num_threads(config.threads);
        const auto plan_seconds = planning.seconds();
        Timer loading;
        MaterialStore material(plan, id, material_file);
        material.claim_execution();
        material.preload();
        const auto load_seconds = loading.seconds();
        Timer offline;
        InputManager inputs(material, plan, input_file, bank_file);
        validate_owned(c, plan, inputs, id);
        std::vector<unsigned> widths{64, 96, 0};
        if (c.carry_cp()) {
            widths.push_back(96);
        }
        MaskedTable band_in(c.mode == "band" ? c.kin : 0, widths),
            band_out(c.mode == "band" ? c.kout : 0, widths);
        std::unique_ptr<Compact> compact;
        if (c.mode != "band") {
            compact = std::make_unique<Compact>(c, material, plan, inputs);
        }
        std::span<const std::uint64_t> omega_in, omega_out;
        if (c.mode == "band") {
            omega_in = inputs.at(named_mask(plan, "omega_in")).share_bits.words();
            omega_out = inputs.at(named_mask(plan, "omega_out")).share_bits.words();
        } else {
            omega_in = compact->omega_in();
            omega_out = compact->omega_out();
        }
        std::unique_ptr<NestedLoop> nl;
        if (c.mode != "compact" && c.algorithm == "nl") {
            nl = std::make_unique<NestedLoop>(c,
                                              material,
                                              plan,
                                              compact ? compact->inbound() : band_in,
                                              compact ? compact->outbound() : band_out,
                                              omega_in,
                                              omega_out);
        }
        std::unique_ptr<DominanceCount> dc;
        if (c.mode != "compact" && c.algorithm == "dc") {
            dc = std::make_unique<DominanceCount>(c,
                                                  material,
                                                  plan,
                                                  compact ? compact->inbound() : band_in,
                                                  compact ? compact->outbound() : band_out,
                                                  omega_in,
                                                  omega_out);
        }
        BitVec neg_in(1), neg_out(1);
        AlignedBuffer<std::uint64_t> nonoverflow(1);
        std::unique_ptr<FusedAnd2A<2, std::uint64_t>> compact_omega;
        std::array<Step *, 1> aggregation_steps{};
        std::unique_ptr<RoundExecutor> aggregate;
        if (c.mode == "compact") {
            compact_omega = std::make_unique<FusedAnd2A<2, std::uint64_t>>(
                material,
                plan,
                named_gate(plan, "output.omega"),
                std::array<std::span<const std::uint64_t>, 2>{neg_in.words(), neg_out.words()},
                nonoverflow.span());
            aggregation_steps[0] = compact_omega.get();
            aggregate = std::make_unique<RoundExecutor>(aggregation_steps);
        }
        OpenLayout output_layout;
        output_layout.add_ring(64, 1);
        if (nl || dc) {
            if (c.output == "count") {
                output_layout.add_ring(64, c.windows.size());
            } else {
                output_layout.add_bits(c.windows.size());
            }
        }
        output_layout.seal();
        AlignedBuffer<std::byte> output_share(output_layout.bytes()),
            output_value(output_layout.bytes());
        std::fill(output_share.span().begin(), output_share.span().end(), std::byte{0});
        std::fill(output_value.span().begin(), output_value.span().end(), std::byte{0});
        OpenBuf output(output_layout, output_share.span(), output_value.span());
        const auto offline_seconds = offline.seconds();
        Net net(id, config);
        std::array<std::byte, 24> context{};
        std::memcpy(context.data(), material.run_id().data(), 16);
        auto hash = plan.fingerprint();
        std::memcpy(context.data() + 16, &hash, 8);
        std::vector<std::array<std::byte, 24>> received(n);
        std::vector<std::span<const std::byte>> tx(n);
        std::vector<std::span<std::byte>> rx(n);
        for (unsigned peer = 0; peer < n; ++peer) {
            if (peer != id) {
                tx[peer] = context;
                rx[peer] = received[peer];
            }
        }
        net.exchange_all(tx, rx);
        for (unsigned peer = 0; peer < n; ++peer) {
            if (peer != id && received[peer] != context) {
                throw std::runtime_error("different preprocessing runs/plans");
            }
        }
        Snapshot input_stats(n), compact_stats(n), nl_stats(n), output_stats(n);
        material.begin_online();
        test::allocations = 0;
        test::count_allocations = true;
        net.reset_stats();
        Timer online_timer;
        Timer timer;
        inputs.run(net);
        input_stats.capture(net, timer.seconds());
        if (c.mode == "band") {
            copy_slot(
                inputs.at(named_mask(plan, "tin")), inputs.at(named_mask(plan, "ain")), band_in);
            copy_slot(
                inputs.at(named_mask(plan, "tout")), inputs.at(named_mask(plan, "aout")), band_out);
            if (c.carry_cp()) {
                for (auto [name, table] : std::array<std::pair<const char *, MaskedTable *>, 2>{
                         {{"cpin", &band_in}, {"cpout", &band_out}}}) {
                    const auto &v = inputs.at(named_mask(plan, name));
                    std::copy(v.delta128.span().begin(),
                              v.delta128.span().end(),
                              table->columns[3].d128.span().begin());
                    std::copy(v.mask128.span().begin(),
                              v.mask128.span().end(),
                              table->columns[3].m128.span().begin());
                }
            }
            if (dc || c.pattern != "P1") {
                for (auto [name, table] : std::array<std::pair<const char *, MaskedTable *>, 2>{
                         {{"vin", &band_in}, {"vout", &band_out}}}) {
                    const auto &value = inputs.at(named_mask(plan, name));
                    std::copy(value.delta_bits.words().begin(),
                              value.delta_bits.words().end(),
                              table->columns[2].bits.words().begin());
                    std::copy(value.mask_bits.words().begin(),
                              value.mask_bits.words().end(),
                              table->columns[2].mask_bits.words().begin());
                }
            }
        }
        if (compact) {
            net.reset_stats();
            timer.reset();
            compact->run(net);
            compact_stats.capture(net, timer.seconds());
        }
        net.reset_stats();
        timer.reset();
        if (nl) {
            nl->run(net);
        } else if (dc) {
#ifdef BJ_GPROF_DC
            moncontrol(1);
#endif
            dc->run(net);
#ifdef BJ_GPROF_DC
            moncontrol(0);
#endif
        } else {
            neg_in.words()[0] = omega_in[0] ^ (id == 0 ? 1 : 0);
            neg_out.words()[0] = (c.pattern == "P3" ? 0 : omega_out[0]) ^ (id == 0 ? 1 : 0);
            aggregate->run(net);
        }
        nl_stats.capture(net, timer.seconds());
        const std::uint64_t omega = nl   ? nl->overflow_share()
                                    : dc ? dc->overflow_share()
                                         : (id == 0 ? 1ull : 0ull) - nonoverflow.data()[0];
        pack_native<std::uint64_t>(64, std::span(&omega, 1), output_share.span().first(8));
        if (nl || dc) {
            const auto &field = output_layout.fields()[1];
            if (c.output == "count") {
                pack_native<std::uint64_t>(64,
                                           nl ? nl->count_shares() : dc->count_shares(),
                                           output_share.span().subspan(field.offset, field.bytes));
            } else {
                store_bits(output_share.span().subspan(field.offset),
                           0,
                           c.windows.size(),
                           nl ? nl->alarm_shares() : dc->alarm_shares());
            }
        }
        net.reset_stats();
        timer.reset();
        net.reveal_to(c.initiator, output);
        timespec receipt{};
        ::clock_gettime(CLOCK_MONOTONIC, &receipt);
        const std::uint64_t ready_ns = std::uint64_t(receipt.tv_sec) * 1000000000 + receipt.tv_nsec;
        const auto total_online = online_timer.seconds();
        output_stats.capture(net, timer.seconds());
        test::count_allocations = false;
        const auto allocations = test::allocations.load();
        const auto &layout = material.layout();
        Json result = {
            {"output_ready_monotonic_ns", ready_ns},
            {"online_seconds", total_online},
            {"id", id},
            {"n", n},
            {"king", c.king},
            {"mode", c.mode},
            {"pattern", c.pattern},
            {"variant", c.variant},
            {"threads", config.threads},
            {"material_mode", material_mode_name(material.mode())},
            {"plan_seconds", plan_seconds},
            {"load_seconds", load_seconds},
            {"offline_seconds", offline_seconds},
            {"input", input_stats.json()},
            {"compact", compact_stats.json()},
            {"nl", nl_stats.json()},
            {"private_output", output_stats.json()},
            {"online_new_calls", allocations},
            {"band_preprocessing_bits", comparison_bits(plan, layout, "band.")},
            {"compact_protocol_preprocessing_bits", comparison_bits(plan, layout, "compact.")},
            {"material_file_bytes", layout.file_bytes},
            {"logical_share_bits", layout.logical_share_bits}};
#ifdef BJ_GPROF_DC
        result["profile_pid"] = ::getpid();
#endif
        if (dc) {
            const auto &d = dc->stats();
            Json sorts = Json::array();
            for (const auto &sort : d.sorts) {
                sorts.push_back({{"comparisons", sort.comparisons},
                                 {"capacity", sort.capacity},
                                 {"depth", sort.depth}});
            }
            result["dc"] = nl_stats.json();
            auto &j = result["dc"];
            j["sorts"] = sorts;
            j["logical_rounds"] = d.rounds;
            j["shuffle_rounds"] = d.shuffle_rounds;
            j["levels"] = d.levels;
            j["time_sort_seconds"] = d.time_sort_seconds;
            j["amount_sort_seconds"] = d.amount_sort_seconds;
            j["levels_seconds"] = d.levels_seconds;
            j["shuffle_seconds"] = d.shuffle_seconds;
            j["requested_pool_factor"] = c.dc_pool_factor;
            j["safe_pool"] = c.dc_pool_safe;
            std::uint64_t mul = 0, masks = 0;
            for (std::size_t i = 0; i < plan.multiplications.size(); ++i) {
                for (auto seg : layout.multiplications[i]) {
                    mul += layout.segments[seg].bits;
                }
            }
            for (std::size_t i = 0; i < plan.masks.size(); ++i) {
                if ((plan.masks[i].name.starts_with("dc.") ||
                     plan.masks[i].name.starts_with("pdc.")) &&
                    plan.masks[i].distribute) {
                    masks += layout.segments[layout.masks[i]].bits;
                }
            }
            std::uint64_t bitmul_bits = 0;
            for (const auto &ids : layout.bitmuls) {
                for (auto id : ids) {
                    bitmul_bits += layout.segments[id].bits;
                }
            }
            const auto reserved = comparison_bits(plan, layout, "dc.") +
                                  comparison_bits(plan, layout, "pdc.") +
                                  comparison_bits(plan, layout, "pattern.") + mul + bitmul_bits;
            std::uint64_t unused = 0;
            for (unsigned r = 0; r < 3; ++r) {
                unused += (d.sorts[r].capacity - d.sorts[r].comparisons) * (r ? 3143ull : 1983ull);
            }
            if (c.pattern != "P1") {
                unused = 0;
                const std::array<std::pair<std::string, SortStats>, 6> extra{
                    {{"pdc.time", d.sorts[0]},
                     {"pdc.amount.0", d.sorts[1]},
                     {"pdc.amount.1", d.sorts[2]},
                     {"pdc.group", d.counterparty_sort},
                     {c.pattern == "P3" ? "pdc.restore" : "pdc.restore.0", d.restore_sorts[0]},
                     {"pdc.restore.1", d.restore_sorts[1]}}};
                for (const auto &[name, stat] : extra) {
                    if (stat.capacity) {
                        const auto id = named_comparison(plan, name + ".compare");
                        const auto &seg = layout.comparisons[id];
                        std::uint64_t bits = 0;
                        for (auto v : seg.tables) {
                            bits += layout.segments[v].bits;
                        }
                        for (auto v : seg.aux) {
                            bits += layout.segments[v].bits;
                        }
                        for (const auto &row : seg.evaluations) {
                            for (auto v : row) {
                                bits += layout.segments[v].bits;
                            }
                        }
                        unused += (stat.capacity - stat.comparisons) * (bits / stat.capacity);
                    }
                }
                auto sort_json = [](const SortStats &x) {
                    return Json{{"comparisons", x.comparisons},
                                {"capacity", x.capacity},
                                {"depth", x.depth}};
                };
                j["counterparty_sort"] = sort_json(d.counterparty_sort);
                j["restore_sorts"] = {sort_json(d.restore_sorts[0]), sort_json(d.restore_sorts[1])};
            }
            j["preprocessing_provisioned_bits"] = reserved + masks;
            j["preprocessing_consumed_bits"] = reserved - unused + masks;
            j["unused_comparison_bits"] = unused;
            j["positional_mask_bits"] = masks;
            std::uint64_t shuffle_bits = 0;
            for (const auto &spec : plan.shuffles) {
                if (spec.name.starts_with("dc.") || spec.name.starts_with("pdc.")) {
                    for (auto ref : spec.inputs) {
                        const auto width = ref < plan.inputs.size()
                                               ? plan.inputs[ref].width
                                               : plan.masks[ref - plan.inputs.size()].width;
                        shuffle_bits += spec.count * (width ? width : 1);
                    }
                }
            }
            j["shuffle_table_bits"] = shuffle_bits;
            j["online_logical_bits"] = d.contribution_bits + shuffle_bits;
        }
        {
            std::uint64_t bits =
                comparison_bits(plan, layout, "band.") + comparison_bits(plan, layout, "pattern.") +
                comparison_bits(plan, layout, "pdc.") + comparison_bits(plan, layout, "dc.") +
                comparison_bits(plan, layout, "output.");
            for (const auto &ids : layout.multiplications) {
                for (auto id : ids) {
                    bits += layout.segments[id].bits;
                }
            }
            for (const auto &ids : layout.bitmuls) {
                for (auto id : ids) {
                    bits += layout.segments[id].bits;
                }
            }
            result["operator_preprocessing_bits"] = bits;
        }
        if (compact) {
            const auto &stats = compact->stats();
            result["compact"]["logical_rounds"] = stats.logical_rounds;
            result["compact"]["shuffle_rounds"] = n;
            result["compact"]["total_network_hops"] = stats.hops + n;
            result["compact"]["shuffle_table_bytes"] = stats.shuffle_bytes;
            result["compact"]["step_seconds"] = stats.seconds;
            std::uint64_t flags = 0, shuffle_bits = 0;
            for (std::size_t i = 0; i < plan.masks.size(); ++i) {
                if (plan.masks[i].name.starts_with("table.f")) {
                    flags += layout.segments[layout.masks[i]].bits;
                }
                if (plan.masks[i].name.starts_with("shuffle.")) {
                    shuffle_bits += layout.segments[layout.masks[i]].bits;
                }
            }
            result["compact_base_preprocessing_bits"] =
                comparison_bits(plan, layout, "compact.") + flags;
            for (auto sid : layout.shuffles[0].permutations) {
                if (layout.segments[sid].storage == SegmentStorage::Dense) {
                    shuffle_bits += layout.segments[sid].bits;
                }
            }
            for (const auto &row : layout.shuffles[0].corrections) {
                for (auto sid : row) {
                    if (layout.segments[sid].storage == SegmentStorage::Dense) {
                        shuffle_bits += layout.segments[sid].bits;
                    }
                }
            }
            result["shuffle_preprocessing_bits"] = shuffle_bits;
        }
        result["local_benchmarks"] = Json::array();
        if (local_repeats) {
            for (unsigned threads : {1u, config.threads}) {
                omp_set_num_threads(static_cast<int>(threads));
                if (compact) {
                    compact->evaluate_local();
                    Timer local;
                    for (unsigned repeat = 0; repeat < local_repeats; ++repeat) {
                        compact->evaluate_local();
                    }
                    result["local_benchmarks"].push_back(
                        {{"kernel", "compact_eq"},
                         {"threads", threads},
                         {"instances_per_second",
                          double(c.size()) * local_repeats / local.seconds()}});
                }
                if (nl && c.pattern == "P1") {
                    nl->evaluate_local();
                    Timer local;
                    for (unsigned repeat = 0; repeat < local_repeats; ++repeat) {
                        nl->evaluate_local();
                    }
                    result["local_benchmarks"].push_back(
                        {{"kernel", "band_pair"},
                         {"threads", threads},
                         {"instances_per_second",
                          double(c.pairs()) * local_repeats / local.seconds()}});
                }
            }
        }
        if (id == c.initiator) {
            Json values = Json::array();
            if (nl || dc) {
                if (c.output == "count") {
                    const auto &field = output_layout.fields()[1];
                    for (std::size_t w = 0; w < c.windows.size(); ++w) {
                        values.push_back(
                            load_packed(output_value.span().subspan(field.offset), w * 64, 64));
                    }
                } else {
                    const auto &field = output_layout.fields()[1];
                    for (std::size_t w = 0; w < c.windows.size(); ++w) {
                        values.push_back(
                            load_packed(output_value.span().subspan(field.offset), w, 1));
                    }
                }
            }
            result["result"] = {{"omega", load_packed(output_value.span(), 0, 64)},
                                {c.output == "alarm" ? "alarm" : "count", values}};
        }
        rusage usage{};
        ::getrusage(RUSAGE_SELF, &usage);
        result["peak_rss_kib"] = usage.ru_maxrss;
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
        std::fprintf(stderr, "investigation_bench: %s\n", e.what());
        return 1;
    }
}
