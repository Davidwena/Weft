#include "bj/ops/quicksort.hpp"
#include "bj/plan/investigation.hpp"

namespace bj {
    namespace {
        struct Builder {
            const InvestigationConfig &c;
            MaterialPlan &p;

            std::size_t mask(const std::string &name,
                             unsigned k,
                             std::size_t n,
                             bool fresh = true,
                             std::vector<MaskTerm> terms = {}) {
                p.masks.push_back({name, k, n, fresh, fresh, std::move(terms)});
                return p.inputs.size() + p.masks.size() - 1;
            }

            void cmp(const std::string &name,
                     CompareKind kind,
                     unsigned k,
                     std::size_t n,
                     std::vector<std::size_t> refs = {},
                     unsigned eval = 1) {
                CompareSpec v;
                v.name = name;
                v.kind = kind;
                v.width = k;
                v.count = n;
                v.inputs = std::move(refs);
                v.evaluations = eval;
                p.comparisons.push_back(v);
            }

            void gate(const std::string &name, unsigned m, unsigned k, std::size_t n) {
                p.gates.push_back({name, m, k, n, {}});
            }

            std::string slot(const std::string &name) const {
                return c.mode == "band" ? name : "slot." + name;
            }

            void sort(const std::string &name,
                      std::size_t N,
                      const std::vector<unsigned> &widths,
                      unsigned keywidth) {
                std::vector<std::size_t> input, output;
                for (std::size_t j = 0; j < widths.size(); ++j) {
                    input.push_back(mask(name + ".mask." + std::to_string(j), widths[j], N));
                    output.push_back(mask(name + ".out." + std::to_string(j), widths[j], N));
                }
                p.shuffles.push_back({name, N, input, output});
                cmp(name + ".compare",
                    CompareKind::MsbFresh,
                    keywidth,
                    dc_pool_capacity(N, c.dc_pool_factor, c.dc_pool_safe));
            }

            void post() {
                const auto Ky = c.y_size();
                if (c.pattern == "P2" || c.pattern == "P3") {
                    cmp("pattern.degree", CompareKind::MsbFresh, 32, Ky);
                    if (c.pattern == "P2") {
                        p.bitmuls.push_back(
                            {"pattern.amount", 64, Ky, named_mask(p, slot("aout"))});
                    } else {
                        gate("pattern.filter", c.algorithm == "dc" ? 2 : 3, 64, Ky);
                    }
                }
                if (c.output == "alarm") {
                    cmp("pattern.alarm", CompareKind::MsbFresh, c.pattern == "P2" ? 64 : 32, 1);
                }
            }

            void slot_amount() {
                const auto source = named_mask(p, slot("ain"));
                auto id = mask("pattern.amount.mask",
                               64,
                               c.kin,
                               false,
                               {{source, 0, 0, c.kin, 1, c.kin ? c.kin : 1, 1}});
                const auto evaluations = c.amount_lower == c.amount_upper ? 1u : 2u;
                cmp("pattern.amount.interval", CompareKind::Interval, 64, c.kin, {id}, evaluations);
            }

            void nl() {
                if (c.pattern == "P3") {
                    slot_amount();
                    const auto K = c.kin, Ky = c.y_size(), P = c.pairs();
                    auto id = mask(
                        "pair.time",
                        c.kt,
                        P,
                        false,
                        {{named_mask(p, slot("tin")), c.p3_query_begin, 0, P, 1, Ky ? Ky : 1, 1},
                         {named_mask(p, slot("tin")), 0, 0, P, Ky ? Ky : 1, K ? K : 1, -1}});
                    cmp("pattern.time.interval",
                        CompareKind::Interval,
                        c.kt,
                        P,
                        {id},
                        c.windows[0].upper ? 2 : 2);
                    gate("pattern.pair", 3, 64, P);
                    return;
                }
                const auto Kx = c.x_size(), Ky = c.y_size(), P = c.pairs();
                const auto tx = named_mask(p, slot(c.pattern == "P4" ? "tout" : "tin")),
                           ty = named_mask(p, slot(c.pattern == "P4" ? "tin" : "tout")),
                           ax = named_mask(p, slot(c.pattern == "P4" ? "aout" : "ain")),
                           ay = named_mask(p, slot(c.pattern == "P4" ? "ain" : "aout"));
                auto terms = [&](std::size_t x, std::int64_t cx, std::size_t y, std::int64_t cy) {
                    return std::vector<MaskTerm>{{x, 0, 0, P, Ky ? Ky : 1, Kx ? Kx : 1, cx},
                                                 {y, 0, 0, P, 1, Ky ? Ky : 1, cy}};
                };
                auto t = mask("pair.time", c.kt, P, false, terms(tx, -1, ty, 1));
                auto a = mask("pair.amount1",
                              c.ka,
                              P,
                              false,
                              terms(ax,
                                    static_cast<std::int64_t>(c.q + c.p),
                                    ay,
                                    -static_cast<std::int64_t>(c.q)));
                auto b = mask("pair.amount2",
                              c.ka,
                              P,
                              false,
                              terms(ax,
                                    -static_cast<std::int64_t>(c.q - c.p),
                                    ay,
                                    static_cast<std::int64_t>(c.q)));
                cmp("band.interval", CompareKind::Interval, c.kt, P, {t}, 2);
                cmp("band.amount1", CompareKind::MsbKnown, c.ka, P, {a});
                cmp("band.amount2", CompareKind::MsbKnown, c.ka, P, {b});
                if (c.pattern == "P4") {
                    auto cp = mask(
                        "pair.cp",
                        64,
                        P,
                        false,
                        terms(named_mask(p, slot("cpout")), -1, named_mask(p, slot("cpin")), 1));
                    cmp("band.cp", CompareKind::Eq, 64, P, {cp});
                }
                gate("band.product.0", c.pattern == "P4" ? 3 : 2, 64, P);
            }

            void dc() {
                const auto Kx = c.x_size(), Ky = c.y_size(), N = Kx + 2 * Ky;
                const auto h = static_cast<unsigned>(std::bit_width(N ? N - 1 : 0));
                if (!N || (c.pattern != "P3" && (!Kx || !Ky))) {
                    return;
                }
                if (c.pattern == "P3") {
                    slot_amount();
                    gate("pdc.weight", 2, 64, c.kin);
                    sort("pdc.time", N, {64, 96, 64}, 64);
                    sort("pdc.restore", N, {96, 64}, std::max(2u, h + 1));
                    return;
                }
                if (c.pattern == "P4") {
                    sort("pdc.group", N, {96, 64, 96, 96, 96, 0, 0, 0, 96}, 96);
                    cmp("pdc.group.eq", CompareKind::EqFresh, 64, N - 1);
                    gate("pdc.group.boundary", 1, 96, N);
                }
                std::vector<unsigned> initial{64, 96, 96, 96, 0, 0, 0};
                if (c.pattern == "P4") {
                    initial.push_back(96);
                }
                sort("pdc.time", N, initial, 64);
                const auto money = 72u;
                std::vector<unsigned> amount(5 + h, 0);
                amount[0] = money;
                amount[1] = 96;
                if (c.pattern == "P2") {
                    amount.push_back(64);
                }
                for (unsigned r = 0; r < 2; ++r) {
                    sort("pdc.amount." + std::to_string(r), N, amount, money);
                }
                for (unsigned ell = 0; ell < h; ++ell) {
                    for (unsigned r = 0; r < 2; ++r) {
                        auto name = "pdc.level." + std::to_string(ell) + "." + std::to_string(r);
                        for (unsigned j = 0; j < 3; ++j) {
                            gate(name + ".and." + std::to_string(j), 2, 64, N);
                        }
                        gate(name + ".b32", 1, 32, N);
                        if (c.pattern == "P2") {
                            gate(name + ".b64", 1, 64, N);
                        }
                        p.multiplications.push_back({name + ".mul64", 64, N});
                        if (ell) {
                            p.multiplications.push_back({name + ".mul32", 32, N});
                            std::vector<unsigned> widths(4 + ell, 0);
                            widths[0] = 32;
                            if (c.pattern == "P2") {
                                widths.push_back(96);
                                widths.push_back(64);
                            }
                            std::vector<std::size_t> in, out;
                            for (std::size_t j = 0; j < widths.size(); ++j) {
                                in.push_back(
                                    mask(name + ".mask." + std::to_string(j), widths[j], N));
                                out.push_back(
                                    mask(name + ".out." + std::to_string(j), widths[j], N));
                            }
                            p.shuffles.push_back({name, N, in, out});
                        }
                    }
                }
                if (c.pattern == "P2") {
                    for (unsigned r = 0; r < 2; ++r) {
                        sort("pdc.restore." + std::to_string(r), N, {96, 64}, std::max(2u, h + 1));
                    }
                }
            }
        };
    }

    void compile_pattern_material(const InvestigationConfig &c, MaterialPlan &p) {
        Builder b{c, p};
        b.post();
        if (c.algorithm == "nl") {
            b.nl();
        } else {
            b.dc();
        }
    }
}
