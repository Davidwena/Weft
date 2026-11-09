#include "bj/ref/investigate.hpp"
#include <algorithm>
#include <numeric>

namespace bj::ref {
    namespace {
        struct Event {
            std::int64_t time;
            std::size_t query;
            int sign;
        };

        std::vector<std::uint64_t> degrees(std::span<const PlainRow> xs,
                                           std::span<const PlainRow> ys,
                                           const InvestigationConfig &c) {
            std::vector<std::uint64_t> answer(ys.size(), 0);
            if (xs.empty() || ys.empty()) {
                return answer;
            }
            std::vector<const PlainRow *> points;
            std::vector<std::uint64_t> amounts;
            points.reserve(xs.size());
            amounts.reserve(xs.size());
            for (const auto &x : xs) {
                points.push_back(&x);
                amounts.push_back(static_cast<std::uint64_t>(x.amount));
            }
            std::sort(points.begin(), points.end(), [](auto *a, auto *b) {
                return a->time < b->time;
            });
            std::sort(amounts.begin(), amounts.end());
            amounts.erase(std::unique(amounts.begin(), amounts.end()), amounts.end());
            std::vector<std::uint64_t> tree(amounts.size() + 1, 0);
            std::vector<Event> events;
            events.reserve(2 * ys.size());
            const auto w = c.windows[0];
            for (std::size_t j = 0; j < ys.size(); ++j) {
                const auto t = static_cast<std::int64_t>(ys[j].time);
                events.push_back({t - static_cast<std::int64_t>(w.lower), j, 1});
                events.push_back({t - static_cast<std::int64_t>(w.upper) - 1, j, -1});
            }
            std::sort(events.begin(), events.end(), [](auto &a, auto &b) {
                return a.time < b.time;
            });
            auto prefix = [&](std::size_t i) {
                std::uint64_t s = 0;
                while (i) {
                    s += tree[i];
                    i -= i & (~i + 1);
                }
                return s;
            };
            std::size_t next = 0;
            for (const auto &e : events) {
                while (next < points.size() &&
                       static_cast<std::int64_t>(points[next]->time) <= e.time) {
                    auto pos =
                        static_cast<std::size_t>(
                            std::lower_bound(amounts.begin(),
                                             amounts.end(),
                                             static_cast<std::uint64_t>(points[next]->amount)) -
                            amounts.begin()) +
                        1;
                    while (pos < tree.size()) {
                        ++tree[pos];
                        pos += pos & (~pos + 1);
                    }
                    ++next;
                }
                const auto value = ys[e.query].amount * c.q;
                const auto low = (value + c.q + c.p - 1) / (c.q + c.p);
                const auto high = value / (c.q - c.p);
                const auto lo = static_cast<std::uint64_t>(std::min(low, uint128_t{UINT64_MAX})),
                           hi = static_cast<std::uint64_t>(std::min(high, uint128_t{UINT64_MAX}));
                const auto l = static_cast<std::size_t>(
                               std::lower_bound(amounts.begin(), amounts.end(), lo) -
                               amounts.begin()),
                           r = static_cast<std::size_t>(
                               std::upper_bound(amounts.begin(), amounts.end(), hi) -
                               amounts.begin());
                const auto count = low > UINT64_MAX || low > high ? 0 : prefix(r) - prefix(l);
                if (e.sign > 0) {
                    answer[e.query] += count;
                } else {
                    answer[e.query] -= count;
                }
            }
            return answer;
        }
    }

    std::string decimal(uint128_t v) {
        std::string s;
        do {
            s.push_back(static_cast<char>('0' + v % 10));
            v /= 10;
        } while (v);
        std::reverse(s.begin(), s.end());
        return s;
    }

    uint128_t aggregate(std::span<const PlainRow> in,
                        std::span<const PlainRow> out,
                        const InvestigationConfig &c) {
        if (c.windows.size() != 1 || !c.q || c.p >= c.q ||
            (c.pattern == "P3" && c.min_cluster < 2)) {
            throw std::invalid_argument("reference requires one valid band window");
        }
        uint128_t result = 0;
        if (c.pattern == "P3") {
            std::vector<std::uint64_t> times;
            for (const auto &x : in) {
                if (x.amount >= c.amount_lower && x.amount < c.amount_upper) {
                    times.push_back(x.time);
                }
            }
            std::sort(times.begin(), times.end());
            for (auto t : times) {
                const auto lower = t > c.windows[0].upper ? t - c.windows[0].upper : 0;
                const auto lo = std::lower_bound(times.begin(), times.end(), lower),
                           hi = std::upper_bound(times.begin(), times.end(), t);
                const auto degree = static_cast<std::size_t>(hi - lo) - 1;
                result += degree >= c.min_cluster - 1;
            }
            return result;
        }
        if (c.pattern == "P4") {
            std::vector<PlainRow> xs(out.begin(), out.end()), ys(in.begin(), in.end());
            auto order = [](auto &a, auto &b) {
                return a.counterparty < b.counterparty;
            };
            std::sort(xs.begin(), xs.end(), order);
            std::sort(ys.begin(), ys.end(), order);
            std::size_t x = 0, y = 0;
            while (x < xs.size() && y < ys.size()) {
                if (xs[x].counterparty < ys[y].counterparty) {
                    ++x;
                    continue;
                }
                if (xs[x].counterparty > ys[y].counterparty) {
                    ++y;
                    continue;
                }
                auto xe = x + 1, ye = y + 1;
                while (xe < xs.size() && xs[xe].counterparty == xs[x].counterparty) {
                    ++xe;
                }
                while (ye < ys.size() && ys[ye].counterparty == ys[y].counterparty) {
                    ++ye;
                }
                for (auto d : degrees(
                         std::span(xs).subspan(x, xe - x), std::span(ys).subspan(y, ye - y), c)) {
                    result += d;
                }
                x = xe;
                y = ye;
            }
            return result;
        }
        const auto d = degrees(in, out, c);
        for (std::size_t y = 0; y < d.size(); ++y) {
            result += c.pattern == "P2" ? (d[y] ? out[y].amount : uint128_t{0}) : uint128_t(d[y]);
        }
        return result;
    }

    ReferenceResult
    evaluate(const Plan &p, std::span<const FilteredRow> rows, std::uint64_t subject) {
        p.validate();
        ReferenceResult result;
        std::vector<const FilteredRow *> full_in, full_out, retained_in, retained_out;
        const bool all = std::any_of(p.templates.begin(), p.templates.end(), [](auto &q) {
            return q.filter.empty();
        });
        std::uint16_t union_mask = 0;
        for (const auto &q : p.templates) {
            if (!q.filter.empty()) {
                const auto i =
                    std::find(p.common.filters.begin(), p.common.filters.end(), q.filter) -
                    p.common.filters.begin();
                union_mask |= static_cast<std::uint16_t>(1u << i);
            }
        }
        for (const auto &x : rows) {
            if (x.row.subject == subject && (all || (x.filters & union_mask))) {
                auto &full = x.row.outgoing ? full_out : full_in;
                auto &retained = x.row.outgoing ? retained_out : retained_in;
                const auto cap = x.row.outgoing ? p.common.kout : p.common.kin;
                full.push_back(&x);
                if (retained.size() < cap) {
                    retained.push_back(&x);
                }
            }
        }
        result.full_in = full_in.size();
        result.full_out = full_out.size();
        result.retained_in = retained_in.size();
        result.retained_out = retained_out.size();
        result.overflow =
            result.full_in > p.common.kin || (std::any_of(p.templates.begin(),
                                                          p.templates.end(),
                                                          [](const auto &q) {
                                                              return q.config.pattern != "P3";
                                                          }) &&
                                              result.full_out > p.common.kout);
        for (const auto &query : p.templates) {
            std::uint16_t bit = 0;
            if (!query.filter.empty()) {
                bit = static_cast<std::uint16_t>(1u << (std::find(p.common.filters.begin(),
                                                                  p.common.filters.end(),
                                                                  query.filter) -
                                                        p.common.filters.begin()));
            }
            auto filtered = [&](const auto &source) {
                std::vector<PlainRow> out;
                for (auto *x : source) {
                    if (!bit || (x->filters & bit)) {
                        out.push_back(x->row);
                    }
                }
                return out;
            };
            const auto fi = filtered(full_in), fo = filtered(full_out), ri = filtered(retained_in),
                       ro = filtered(retained_out);
            TemplateResult row;
            row.name = query.name;
            row.full = aggregate(fi, fo, query.config);
            row.retained = aggregate(ri, ro, query.config);
            row.full_alarm = row.full >= query.config.threshold;
            row.retained_alarm = row.retained >= query.config.threshold;
            if (row.retained > row.full || (!result.overflow && row.retained != row.full)) {
                throw std::logic_error("reference lower-bound/equality invariant");
            }
            result.templates.push_back(row);
        }
        return result;
    }
}
