#include "bj/ops/quicksort.hpp"
#include "bj/plan/investigation.hpp"
#include <cmath>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <limits>

namespace bj {

    std::size_t dc_pool_capacity(std::size_t n, double factor, bool safe) {
        (void)safe;
        if (!std::isfinite(factor) || factor <= 0 || factor > 100 || n >= (std::size_t{1} << 31)) {
            throw std::invalid_argument("DC pool dimensions");
        }
        if (n < 2) {
            return 0;
        }

        if (n <= 64) {
            using boost::multiprecision::cpp_rational;
            cpp_rational H = 0;
            for (std::size_t i = 1; i <= n; ++i) {
                H += cpp_rational(1) / i;
            }
            const cpp_rational value = cpp_rational(factor) * (2 * (n + 1) * H - 4 * n);
            const auto numerator = boost::multiprecision::numerator(value);
            const auto denominator = boost::multiprecision::denominator(value);
            return ((numerator + denominator - 1) / denominator).convert_to<std::size_t>();
        }
        long double harmonic = 0, correction = 0;
        for (std::size_t i = 1; i <= n; ++i) {
            const auto term = 1.L / i - correction, next = harmonic + term;
            correction = (next - harmonic) - term;
            harmonic = next;
        }
        const long double value = factor * (2.L * (n + 1) * harmonic - 4.L * n);

        const auto error =
            64 * std::numeric_limits<long double>::epsilon() * std::max(1.L, std::abs(value));
        if (std::abs(value - std::round(value)) <= error) {
            using Wide = boost::multiprecision::cpp_dec_float_100;
            Wide H = 0;
            for (std::size_t i = 1; i <= n; ++i) {
                H += Wide(1) / i;
            }
            return boost::multiprecision::ceil(Wide(factor) * (2 * (n + 1) * H - 4 * n))
                .convert_to<std::size_t>();
        }
        return static_cast<std::size_t>(std::ceil(value));
    }

    void compile_dc_material(const InvestigationConfig &c, MaterialPlan &p) {
        const auto N = c.kin + 2 * c.kout;
        const auto h = static_cast<unsigned>(std::bit_width(N ? N - 1 : 0));
        if (!c.kin || !c.kout) {
            return;
        }
        auto add = [&](const std::string &name,
                       unsigned k,
                       bool fresh = true,
                       std::vector<MaskTerm> terms = {}) {
            p.masks.push_back({name, k, N, fresh, fresh, std::move(terms)});
            return p.inputs.size() + p.masks.size() - 1;
        };
        auto term = [&](const std::string &name,
                        std::size_t dst,
                        std::size_t len,
                        std::int64_t coefficient) {
            return MaskTerm{named_mask(p, name), 0, dst, len, 1, len ? len : 1, coefficient};
        };
        auto slot = [&](const std::string &name) {
            return c.mode == "band" ? name : "slot." + name;
        };
        std::vector<std::size_t> initial;
        for (unsigned k = 0; k < 3; ++k) {
            std::vector<MaskTerm> t;
            if (k == 0) {
                t.push_back(term(slot("tin"), 0, c.kin, std::int64_t{3} << h));
                t.push_back(term(slot("tout"), c.kin, c.kout, std::int64_t{3} << h));
                t.push_back(term(slot("tout"), c.kin + c.kout, c.kout, std::int64_t{3} << h));
            } else {
                t.push_back(
                    term(slot("ain"),
                         0,
                         c.kin,
                         static_cast<std::int64_t>((3 * (k == 1 ? c.q - c.p : c.q + c.p)) << h)));
                t.push_back(
                    term(slot("aout"), c.kin, c.kout, static_cast<std::int64_t>((3 * c.q) << h)));
                t.push_back(term(slot("aout"),
                                 c.kin + c.kout,
                                 c.kout,
                                 static_cast<std::int64_t>((3 * c.q) << h)));
            }
            initial.push_back(add("dc.initial." + std::to_string(k), k ? 72 : 64, false, t));
        }
        initial.push_back(add("dc.initial.3", 72, false));
        for (unsigned k = 0; k < 3; ++k) {
            const auto point = k == 0;
            const auto dest = point ? 0 : c.kin + (k == 2 ? c.kout : 0);
            const auto len = point ? c.kin : c.kout;
            initial.push_back(add("dc.initial." + std::to_string(k + 4),
                                  0,
                                  false,
                                  {term(slot(point ? "vin" : "vout"), dest, len, 1)}));
        }
        auto shuffle = [&](const std::string &name, const std::vector<std::size_t> &in) {
            std::vector<std::size_t> out;
            for (unsigned j = 0; j < in.size(); ++j) {
                const auto width = p.masks.at(in[j] - p.inputs.size()).width;
                out.push_back(add(name + ".out." + std::to_string(j), width));
            }
            p.shuffles.push_back({name, N, in, out});
        };
        shuffle("dc.time", initial);
        const auto C = dc_pool_capacity(N, c.dc_pool_factor, c.dc_pool_safe);
        for (unsigned r = 0; r < 3; ++r) {
            CompareSpec s;
            s.name = "dc.sort." + std::to_string(r);
            s.kind = CompareKind::MsbFresh;
            s.width = r ? 72 : 64;
            s.count = C;
            p.comparisons.push_back(s);
        }
        for (unsigned r = 0; r < 2; ++r) {
            std::vector<std::size_t> in;
            for (unsigned j = 0; j < 5 + h; ++j) {
                in.push_back(add("dc.amount." + std::to_string(r) + ".mask." + std::to_string(j),
                                 j < 2 ? 72 : 0));
            }
            shuffle("dc.amount." + std::to_string(r), in);
        }
        for (unsigned level = 0; level < h; ++level) {
            for (unsigned r = 0; r < 2; ++r) {
                const auto prefix = "dc.level." + std::to_string(level) + "." + std::to_string(r);
                for (unsigned j = 0; j < 3; ++j) {
                    p.gates.push_back({prefix + ".and." + std::to_string(j), 2, 64, N, {}});
                }
                p.gates.push_back({prefix + ".b2a", 1, 32, N, {}});
                p.multiplications.push_back({prefix + ".mul64", 64, N});
                if (level) {
                    p.multiplications.push_back({prefix + ".mul32", 32, N});
                    std::vector<std::size_t> in;
                    for (unsigned j = 0; j < 4 + level; ++j) {
                        in.push_back(add(prefix + ".mask." + std::to_string(j), j == 0 ? 32 : 0));
                    }
                    shuffle(prefix, in);
                }
            }
        }
        if (c.output == "alarm") {
            CompareSpec s;
            s.name = "dc.alarm";
            s.kind = CompareKind::MsbFresh;
            s.width = 32;
            s.count = 1;
            p.comparisons.push_back(s);
        }
    }
}
