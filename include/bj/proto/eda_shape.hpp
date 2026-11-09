#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>

namespace bj {

    struct EdaGate {
        unsigned a, b, out;
        int xor_node = -1;
    };

    struct EdaShape {
        unsigned k, nodes, result;
        bool equality;
        std::vector<std::vector<EdaGate>> layers;
        std::size_t gates = 0;

        explicit EdaShape(unsigned bits, bool eq)
            : k(bits), nodes(2 * bits), result(0), equality(eq) {
            if (!k || k > 72) {
                throw std::invalid_argument("eda width must be 1..72");
            }
            std::vector<unsigned> p(k), g(k);
            for (unsigned i = 0; i < k; ++i) {
                p[i] = i;
                g[i] = k + i;
            }
            if (eq) {
                while (p.size() > 1) {
                    layers.emplace_back();
                    std::vector<unsigned> next;
                    for (unsigned i = 0; i + 1 < p.size(); i += 2) {
                        layers.back().push_back({p[i], p[i + 1], nodes++});
                        next.push_back(nodes - 1);
                    }
                    if (p.size() % 2) {
                        next.push_back(p.back());
                    }
                    p = std::move(next);
                }
                result = p[0];
            } else {

                for (unsigned d = 1; d < k - 1; d *= 2) {
                    layers.emplace_back();
                    auto pn = p, gn = g;
                    for (unsigned i = d; i < k - 1; ++i) {
                        layers.back().push_back({p[i], g[i - d], nodes++, static_cast<int>(g[i])});
                        gn[i] = nodes - 1;
                        layers.back().push_back({p[i], p[i - d], nodes++});
                        pn[i] = nodes - 1;
                    }
                    p = std::move(pn);
                    g = std::move(gn);
                }
                result = k == 1 ? 0 : g[k - 2];
            }
            for (const auto &layer : layers) {
                gates += layer.size();
            }
        }
    };
}
