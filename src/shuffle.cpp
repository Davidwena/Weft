#include "bj/core/multi_round.hpp"
#include "bj/proto/shuffle.hpp"

namespace bj {
    Shuffle::Shuffle(MaskedTable &table, MaterialStore &s, const MaterialPlan &p, std::size_t index)
        : table_(table), party_(s.party()), parties_(p.n),
          permutation_(p.shuffles.at(index).count) {
        const auto &spec = p.shuffles[index];
        const auto &ids = s.layout().shuffles[index];
        s.claim_shuffle(index);
        if (table.columns.size() != spec.inputs.size()) {
            throw std::invalid_argument("shuffle table columns");
        }
        const auto perm = s.packed(ids.permutations[party_]);
        if (s.mode() == MaterialMode::Prg) {
            permutation_from_seed(perm, permutation_.span());
        } else {
            unpack_native<std::uint32_t>(32, perm, permutation_.span());
        }

        std::vector<unsigned char> seen(spec.count, 0);
        for (auto position : permutation_.span()) {
            if (position >= spec.count || seen[position]++) {
                throw std::invalid_argument("invalid shuffle permutation");
            }
        }
        std::size_t bytes = 0;
        correction_.reserve(spec.inputs.size());
        output_.reserve(spec.inputs.size());
        for (std::size_t j = 0; j < spec.inputs.size(); ++j) {
            auto &c = table.columns[j];
            if (c.count != spec.count) {
                throw std::invalid_argument("shuffle row count");
            }
            offsets_.push_back(bytes);
            bytes += bits_bytes(checked_bits(c.count, c.width ? c.width : 1));
            correction_.emplace_back(c.width, c.count);
            output_.emplace_back(c.width, c.count);
            load_mask_share(output_.back(), s, p, spec.outputs[j]);
            auto packed = s.packed(ids.corrections[party_][j]);
            auto &corr = correction_.back();
            if (!c.width) {
                corr.bits.unpack(packed);
            } else if (c.width <= 32) {
                unpack_native<std::uint32_t>(c.width, packed, corr.d32.span());
            } else if (c.width <= 64) {
                unpack_native<std::uint64_t>(c.width, packed, corr.d64.span());
            } else {
                unpack_native<uint128_t>(c.width, packed, corr.d128.span());
            }
        }
        received_ = AlignedBuffer<std::byte>(bytes);
        sent_ = AlignedBuffer<std::byte>(bytes);
    }

    void Shuffle::check(Net &net) {
        if (used_ || !table_.dealer_knows_positions ||
            !std::all_of(table_.columns.begin(),
                         table_.columns.end(),
                         [](const auto &c) {
                             return c.dealer_knows_mask;
                         }) ||
            net.id() != party_ || net.size() != parties_) {
            throw std::logic_error(
                "shuffle requires unused material and dealer-known positional masks");
        }
        used_ = true;
    }

    void Shuffle::pack_input() {
        {
            for (std::size_t j = 0; j < table_.columns.size(); ++j) {
                auto &c = table_.columns[j];
                auto bytes = received_.span().subspan(
                    offsets_[j], bits_bytes(c.count * (c.width ? c.width : 1)));
                if (!c.width) {
                    c.bits.pack(bytes);
                } else if (c.width <= 32) {
                    pack_native<std::uint32_t>(c.width, c.d32.span(), bytes);
                } else if (c.width <= 64) {
                    pack_native<std::uint64_t>(c.width, c.d64.span(), bytes);
                } else {
                    pack_native<uint128_t>(c.width, c.d128.span(), bytes);
                }
            }
        }
    }

    void Shuffle::transform() {
        for (std::size_t j = 0; j < table_.columns.size(); ++j) {
            auto &src = table_.columns[j];
            auto &dst = output_[j];
            const auto &corr = correction_[j];
            auto packed = received_.span().subspan(
                offsets_[j], bits_bytes(src.count * (src.width ? src.width : 1)));
            auto wire = sent_.span().subspan(offsets_[j], packed.size());
            if (!src.width) {
                src.bits.unpack(packed);
                auto out = dst.bits.words();
#pragma omp parallel for if (src.count >= 16384) schedule(static)
                for (std::size_t word = 0; word < out.size(); ++word) {
                    std::uint64_t bits = 0;
                    for (unsigned lane = 0; lane < 64 && word * 64 + lane < src.count; ++lane) {
                        bits |= std::uint64_t(src.bits.get(permutation_.data()[word * 64 + lane]))
                                << lane;
                    }
                    out[word] = bits ^ corr.bits.words()[word];
                }
                dst.bits.pack(wire);
            } else {
                auto gather = [&]<class T>() {
                    auto in = src.delta<T>(), out = dst.delta<T>();
                    unpack_native<T>(src.width, packed, in);
                    const auto correction = corr.delta<T>();
                    const auto mask = native_mask<T>(src.width);

#pragma omp parallel for if (src.count >= 16384) schedule(static)
                    for (std::size_t tile = 0; tile < (src.count + 4095) / 4096; ++tile)
                        for (std::size_t i = tile * 4096;
                             i < std::min(src.count, (tile + 1) * 4096);
                             ++i)
                            out[i] = (in[permutation_.data()[i]] + correction[i]) & mask;
                    pack_native<T>(src.width, out, wire);
                };
                if (src.width <= 32) {
                    gather.template operator()<std::uint32_t>();
                } else if (src.width <= 64) {
                    gather.template operator()<std::uint64_t>();
                } else {
                    gather.template operator()<uint128_t>();
                }
            }
        }
    }

    void Shuffle::finish(std::span<const std::byte> packed) {
        for (std::size_t j = 0; j < table_.columns.size(); ++j) {
            auto &dst = table_.columns[j];
            const auto &mask = output_[j];
            auto bytes =
                packed.subspan(offsets_[j], bits_bytes(dst.count * (dst.width ? dst.width : 1)));
            if (!dst.width) {
                dst.bits.unpack(bytes);
                std::copy(mask.mask_bits.words().begin(),
                          mask.mask_bits.words().end(),
                          dst.mask_bits.words().begin());
            } else {
                auto finish = [&]<class T>() {
                    unpack_native<T>(dst.width, bytes, dst.delta<T>());
                    std::copy(mask.mask<T>().begin(), mask.mask<T>().end(), dst.mask<T>().begin());
                };
                if (dst.width <= 32) {
                    finish.template operator()<std::uint32_t>();
                } else if (dst.width <= 64) {
                    finish.template operator()<std::uint64_t>();
                } else {
                    finish.template operator()<uint128_t>();
                }
            }
        }
    }

    void Shuffle::run(Net &net) {
        if (auto *h = Net::thread_hook()) {
            Shuffle *self = this;
            h->shuffle(std::span<Shuffle *const>(&self, 1));
            return;
        }
        check(net);
        if (party_ == 0) {
            pack_input();
        } else {
            net.recv(party_ - 1, received_.span());
        }
        transform();
        if (party_ + 1 < parties_) {
            net.send(party_ + 1, sent_.span());
            net.broadcast_from(parties_ - 1, received_.span());
            finish(received_.span());
        } else {
            net.broadcast_from(parties_ - 1, sent_.span());
            finish(sent_.span());
        }
    }

    ShuffleGroup::ShuffleGroup(std::span<Shuffle *const> group)
        : shuffles_(group.begin(), group.end()) {
        if (group.empty()) {
            throw std::invalid_argument("empty shuffle group");
        }
        std::size_t bytes = 0;
        for (std::size_t i = 0; i < group.size(); ++i) {
            if (!group[i]) {
                throw std::invalid_argument("null shuffle");
            }
            for (std::size_t j = 0; j < i; ++j) {
                if (group[i] == group[j]) {
                    throw std::invalid_argument("duplicate shuffle");
                }
            }
            bytes += group[i]->bytes();
        }
        received_ = AlignedBuffer<std::byte>(bytes);
        sent_ = AlignedBuffer<std::byte>(bytes);
    }

    void ShuffleGroup::run(Net &net) {
        if (auto *h = Net::thread_hook()) {
            h->shuffle(shuffles_);
            return;
        }
        for (auto *s : shuffles_) {
            s->check(net);
        }
        if (net.id() == 0) {
            std::size_t off = 0;
            for (auto *s : shuffles_) {
                s->pack_input();
                std::copy(
                    s->received_.span().begin(), s->received_.span().end(), received_.data() + off);
                off += s->bytes();
            }
        } else {
            net.recv(net.id() - 1, received_.span());
        }
        std::size_t off = 0;
        for (auto *s : shuffles_) {
            std::copy_n(received_.data() + off, s->bytes(), s->received_.data());
            s->transform();
            std::copy(s->sent_.span().begin(), s->sent_.span().end(), sent_.data() + off);
            off += s->bytes();
        }
        if (net.id() + 1 < net.size()) {
            net.send(net.id() + 1, sent_.span());
            net.broadcast_from(net.size() - 1, received_.span());
        } else {
            net.broadcast_from(net.size() - 1, sent_.span());
        }
        const auto final = net.id() + 1 == net.size() ? sent_.span() : received_.span();
        off = 0;
        for (auto *s : shuffles_) {
            s->finish(final.subspan(off, s->bytes()));
            off += s->bytes();
        }
    }

    void ShuffleBatch::run(Net &net, std::span<Shuffle *const> group) {
        std::size_t total = 0;
        for (auto *s : group) {
            if (s->bytes() > sent_.size() - total) {
                throw std::length_error("merged shuffle bound");
            }
            total += s->bytes();
            s->check(net);
        }
        auto received = received_.span().first(total), sent = sent_.span().first(total);
        if (net.id() == 0) {
            std::size_t off = 0;
            for (auto *s : group) {
                s->pack_input();
                std::copy(s->received_.span().begin(),
                          s->received_.span().end(),
                          received.begin() + static_cast<std::ptrdiff_t>(off));
                off += s->bytes();
            }
        } else {
            net.recv(net.id() - 1, received);
        }
        std::size_t off = 0;
        for (auto *s : group) {
            std::copy_n(received.data() + off, s->bytes(), s->received_.data());
            s->transform();
            std::copy(s->sent_.span().begin(),
                      s->sent_.span().end(),
                      sent.begin() + static_cast<std::ptrdiff_t>(off));
            off += s->bytes();
        }
        if (net.id() + 1 < net.size()) {
            net.send(net.id() + 1, sent);
            net.broadcast_from(net.size() - 1, received);
        } else {
            net.broadcast_from(net.size() - 1, sent);
        }
        const auto final = net.id() + 1 == net.size() ? sent : received;
        off = 0;
        for (auto *s : group) {
            s->finish(final.subspan(off, s->bytes()));
            off += s->bytes();
        }
    }
}
