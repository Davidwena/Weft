#include "bj/core/input.hpp"
#include "bj/data/bank.hpp"
#include "bj/plan/investigation.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

namespace bj {
    PreparedInput::PreparedInput(const InputSpec &in, unsigned party)
        : spec(in), mask_bits(in.width ? 0 : in.count), delta_bits(in.width ? 0 : in.count),
          share_bits(in.width ? 0 : in.count),
          owner_bits(!in.width && in.owner == party ? in.count : 0),
          value_bits(!in.width && in.owner == party ? in.count : 0) {
        if (in.width && in.width <= 32) {
            mask32 = AlignedBuffer<std::uint32_t>(in.count);
            delta32 = AlignedBuffer<std::uint32_t>(in.count);
            share32 = AlignedBuffer<std::uint32_t>(in.count);
            if (in.owner == party) {
                owner32 = AlignedBuffer<std::uint32_t>(in.count);
                value32 = AlignedBuffer<std::uint32_t>(in.count);
            }
        } else if (in.width && in.width <= 64) {
            mask64 = AlignedBuffer<std::uint64_t>(in.count);
            delta64 = AlignedBuffer<std::uint64_t>(in.count);
            share64 = AlignedBuffer<std::uint64_t>(in.count);
            if (in.owner == party) {
                owner64 = AlignedBuffer<std::uint64_t>(in.count);
                value64 = AlignedBuffer<std::uint64_t>(in.count);
            }
        } else if (in.width > 64) {
            mask128 = AlignedBuffer<uint128_t>(in.count);
            delta128 = AlignedBuffer<uint128_t>(in.count);
            share128 = AlignedBuffer<uint128_t>(in.count);
            if (in.owner == party) {
                owner128 = AlignedBuffer<uint128_t>(in.count);
                value128 = AlignedBuffer<uint128_t>(in.count);
            }
        }
    }

    InputManager::InputManager(MaterialStore &material,
                               const MaterialPlan &plan,
                               const std::filesystem::path &path,
                               const std::filesystem::path &bank_file,
                               std::size_t bank_offset,
                               std::size_t bank_count)
        : party_(material.party()), n_(plan.n), wire_(n_), send_(n_), recv_(n_) {
        std::unique_ptr<BankData> bank;
        if (!bank_file.empty()) {
            bank = std::make_unique<BankData>(bank_file, bank_offset, bank_count);
            if (bank->bank() != party_ || bank->parties() != n_) {
                throw std::invalid_argument("bank input party mismatch");
            }
        }
        nlohmann::json values = nlohmann::json::object();
        if (!path.empty()) {
            std::ifstream file(path);
            if (!file) {
                throw std::runtime_error("cannot open private input");
            }
            values = nlohmann::json::parse(file).at("inputs");
            if (!values.is_object()) {
                throw std::invalid_argument("private input object required");
            }
        }
        for (auto it = values.begin(); it != values.end(); ++it) {
            auto found =
                std::find_if(plan.inputs.begin(), plan.inputs.end(), [&](const InputSpec &in) {
                    return in.name == it.key() && in.owner == party_;
                });
            if (found == plan.inputs.end()) {
                throw std::invalid_argument("private input contains non-owned/unknown field");
            }
        }
        const auto &layout = material.layout();
        std::vector<std::size_t> sizes(n_, 0);
        for (std::size_t i = 0; i < plan.inputs.size(); ++i) {
            const auto &spec = plan.inputs[i];
            material.claim_input(i);
            auto input = std::make_unique<PreparedInput>(spec, party_);
            input->wire_offset = sizes[spec.owner];
            const auto bytes = bits_bytes(checked_bits(spec.count, spec.width ? spec.width : 1));
            if (bytes > SIZE_MAX - sizes[spec.owner]) {
                throw std::length_error("input wire size overflow");
            }
            sizes[spec.owner] += bytes;
            const auto mask = material.packed(layout.input_shares[i]);
            if (!spec.width) {
                input->mask_bits.unpack(mask);
            } else if (spec.width <= 32) {
                unpack_native<std::uint32_t>(spec.width, mask, input->mask32.span());
            } else if (spec.width <= 64) {
                unpack_native<std::uint64_t>(spec.width, mask, input->mask64.span());
            } else {
                unpack_native<uint128_t>(spec.width, mask, input->mask128.span());
            }
            if (spec.owner == party_) {
                const auto full = material.packed(layout.owner_masks[i]);
                if (!spec.width) {
                    input->owner_bits.unpack(full);
                } else if (spec.width <= 32) {
                    unpack_native<std::uint32_t>(spec.width, full, input->owner32.span());
                } else if (spec.width <= 64) {
                    unpack_native<std::uint64_t>(spec.width, full, input->owner64.span());
                } else {
                    unpack_native<uint128_t>(spec.width, full, input->owner128.span());
                }
                const bool loaded = bank && bank->fill(*input);
                if (loaded && values.contains(spec.name)) {
                    throw std::invalid_argument("duplicate binary/JSON bank input");
                }
                if (!loaded) {
                    const auto found = values.find(spec.name);
                    if ((found == values.end() && spec.count) ||
                        (found != values.end() &&
                         (!found->is_array() || found->size() != spec.count))) {
                        throw std::invalid_argument("owned input length/missing field");
                    }
                    for (std::size_t j = 0; j < spec.count; ++j) {
                        const auto &v = (*found)[j];
                        uint128_t x;
                        if (v.is_string() && spec.width > 64) {
                            x = 0;
                            const auto text = v.get<std::string>();
                            if (text.empty()) {
                                throw std::invalid_argument("empty wide input");
                            }
                            for (char ch : text) {
                                if (ch < '0' || ch > '9' ||
                                    x > (native_mask<uint128_t>(spec.width) - (ch - '0')) / 10) {
                                    throw std::invalid_argument("wide input outside range");
                                }
                                x = x * 10 + (ch - '0');
                            }
                        } else if (v.is_boolean() && !spec.width) {
                            x = v.get<bool>();
                        } else if (v.is_number_unsigned()) {
                            x = v.get<std::uint64_t>();
                        } else if (v.is_number_integer() && v.get<std::int64_t>() >= 0) {
                            x = static_cast<std::uint64_t>(v.get<std::int64_t>());
                        } else {
                            throw std::invalid_argument(
                                "input must be a nonnegative integer/Boolean");
                        }
                        if (x > (spec.width ? native_mask<uint128_t>(spec.width) : 1)) {
                            throw std::invalid_argument("input outside public range");
                        }
                        if (!spec.width) {
                            input->value_bits.set(j, x != 0);
                        } else if (spec.width <= 32) {
                            input->value32.data()[j] = static_cast<std::uint32_t>(x);
                        } else if (spec.width <= 64) {
                            input->value64.data()[j] = static_cast<std::uint64_t>(x);
                        } else {
                            input->value128.data()[j] = x;
                        }
                    }
                }
            }
            inputs_.push_back(std::move(input));
        }
        if (!plan.context.empty()) {
            const auto document = nlohmann::json::parse(plan.context, nullptr, false);
            if (document.is_object() && document.contains("investigation")) {
                const auto c = InvestigationConfig::parse(document);
                const auto theta = std::uint64_t{1} << c.bt;
                if (c.mode == "band") {
                    for (auto label : {std::string("in"), std::string("out")}) {
                        const auto &t = at(named_mask(plan, "t" + label));
                        if (t.spec.owner != party_) {
                            continue;
                        }
                        const auto &a = at(named_mask(plan, "a" + label));
                        const auto sentinel = label == "in" ? theta : 0 - theta;
                        const PreparedInput *valid = nullptr;
                        for (std::size_t i = 0; i < plan.inputs.size(); ++i) {
                            if (plan.inputs[i].name == "v" + label) {
                                valid = &at(i);
                            }
                        }
                        const PreparedInput *cp =
                            c.carry_cp() ? &at(named_mask(plan, "cp" + label)) : nullptr;
                        for (std::size_t i = 0; i < t.spec.count; ++i) {
                            const auto time = t.value64.data()[i];
                            const auto amount = a.value128.data()[i];
                            const bool dummy = time == sentinel;
                            if ((!dummy && time >= theta) ||
                                (dummy ? amount != 0
                                       : !amount || amount >= (uint128_t{1} << c.ba))) {
                                throw std::invalid_argument(
                                    "owned time/amount outside query range");
                            }
                            if (valid && valid->value_bits.get(i) == dummy) {
                                throw std::invalid_argument(
                                    "dummy validity inconsistent with sentinel");
                            }
                            if (cp && (cp->value128.data()[i] >= (uint128_t{1} << 64) ||
                                       (dummy && cp->value128.data()[i] != 0))) {
                                throw std::invalid_argument(
                                    "owned counterparty outside query range");
                            }
                        }
                    }
                } else {
                    for (unsigned bank = 0; bank < c.n; ++bank) {
                        if (bank != party_) {
                            continue;
                        }
                        const auto suffix = "_" + std::to_string(bank);
                        const auto &a = at(named_mask(plan, "a" + suffix));
                        const auto &t = at(named_mask(plan, "t" + suffix));
                        for (std::size_t i = 0; i < a.spec.count; ++i) {
                            if (!a.value128.data()[i] ||
                                a.value128.data()[i] >= (uint128_t{1} << c.ba) ||
                                t.value64.data()[i] >= theta) {
                                throw std::invalid_argument("real amount/time outside query range");
                            }
                        }
                        if (c.carry_cp()) {
                            const auto &cp = at(named_mask(plan, "cp" + suffix));
                            for (auto value : cp.value128.span()) {
                                if (value >= (uint128_t{1} << 64)) {
                                    throw std::invalid_argument(
                                        "counterparty outside 64-bit value range");
                                }
                            }
                        }
                    }
                }
            }
        }
        for (unsigned owner = 0; owner < n_; ++owner) {
            wire_[owner] = AlignedBuffer<std::byte>(sizes[owner]);
        }
        for (unsigned peer = 0; peer < n_; ++peer) {
            if (peer != party_) {
                send_[peer] = wire_[party_].span();
                recv_[peer] = wire_[peer].span();
            }
        }
    }

    void InputManager::run(Net &net) {
        if (used_ || net.id() != party_ || net.size() != n_) {
            throw std::logic_error("input material reused/party mismatch");
        }
        used_ = true;
        for (auto &ptr : inputs_) {
            auto &in = *ptr;
            const auto &spec = in.spec;
            if (spec.owner != party_) {
                continue;
            }
            const auto bytes = bits_bytes(checked_bits(spec.count, spec.width ? spec.width : 1));
            auto out = wire_[party_].span().subspan(in.wire_offset, bytes);
            if (!spec.width) {
                const auto a = in.value_bits.words(), b = in.owner_bits.words();
                auto delta = in.delta_bits.words();
#pragma omp parallel for if (delta.size() >= 16384) schedule(static)
                for (std::size_t i = 0; i < delta.size(); ++i) {
                    delta[i] = a[i] ^ b[i];
                }
                in.delta_bits.pack(out);
            } else {
                auto encode = [&]<class T>(std::span<const T> value,
                                           std::span<const T> mask,
                                           std::span<T> delta) {
                    const auto ring_mask = native_mask<T>(spec.width);
#pragma omp parallel for if (delta.size() >= 16384) schedule(static)
                    for (std::size_t i = 0; i < delta.size(); ++i)
                        delta[i] = (value[i] + mask[i]) & ring_mask;
                    pack_native<T>(spec.width, delta, out);
                };
                if (spec.width <= 32) {
                    encode.template operator()<std::uint32_t>(
                        in.value32.span(), in.owner32.span(), in.delta32.span());
                } else if (spec.width <= 64) {
                    encode.template operator()<std::uint64_t>(
                        in.value64.span(), in.owner64.span(), in.delta64.span());
                } else {
                    encode.template operator()<uint128_t>(
                        in.value128.span(), in.owner128.span(), in.delta128.span());
                }
            }
        }
        net.exchange_all(send_, recv_);
        for (auto &ptr : inputs_) {
            auto &in = *ptr;
            const auto &spec = in.spec;
            const auto bytes = bits_bytes(checked_bits(spec.count, spec.width ? spec.width : 1));
            const auto data = wire_[spec.owner].span().subspan(in.wire_offset, bytes);
            if (!spec.width) {
                in.delta_bits.unpack(data);
                auto out = in.share_bits.words();
                const auto delta = in.delta_bits.words(), mask = in.mask_bits.words();
#pragma omp parallel for if (out.size() >= 16384) schedule(static)
                for (std::size_t i = 0; i < out.size(); ++i) {
                    out[i] = mask[i] ^ (party_ == 0 ? delta[i] : 0);
                }
            } else {
                auto decode =
                    [&]<class T>(std::span<T> delta, std::span<const T> mask, std::span<T> out) {
                        unpack_native<T>(spec.width, data, delta);
                        const auto ring_mask = native_mask<T>(spec.width);
#pragma omp parallel for if (out.size() >= 16384) schedule(static)
                        for (std::size_t i = 0; i < out.size(); ++i)
                            out[i] = ((party_ == 0 ? delta[i] : T{0}) - mask[i]) & ring_mask;
                    };
                if (spec.width <= 32) {
                    decode.template operator()<std::uint32_t>(
                        in.delta32.span(), in.mask32.span(), in.share32.span());
                } else if (spec.width <= 64) {
                    decode.template operator()<std::uint64_t>(
                        in.delta64.span(), in.mask64.span(), in.share64.span());
                } else {
                    decode.template operator()<uint128_t>(
                        in.delta128.span(), in.mask128.span(), in.share128.span());
                }
            }
        }
    }
}
