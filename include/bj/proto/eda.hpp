#pragma once
#include "bj/proto/msb.hpp"
#include "bj/proto/eda_shape.hpp"

namespace bj {
    template <class T> class EdaBatch {
        EdaShape shape_;
        unsigned party_;
        std::size_t n_, words_;
        std::span<const T> x_;
        std::span<std::uint64_t> output_;
        AlignedBuffer<T> r_, hat_, send_;
        AlignedBuffer<std::uint64_t> bits_, nodes_, triples_;

        std::span<std::uint64_t> node(unsigned i) {
            return nodes_.span().subspan(i * words_, words_);
        }

        void finish() {
            const auto value = node(shape_.result);
            for (std::size_t w = 0; w < words_; ++w) {
                output_[w] =
                    value[w] ^ (!shape_.equality && shape_.k > 1 ? node(shape_.k - 1)[w] : 0);
            }
        }

        void initialize() {
            for (unsigned b = 0; b < shape_.k; ++b) {
                for (std::size_t w = 0; w < words_; ++w) {
                    const auto take = static_cast<unsigned>(std::min<std::size_t>(64, n_ - w * 64));
                    std::uint64_t c = 0;
                    for (unsigned lane = 0; lane < take; ++lane) {
                        c |= std::uint64_t((hat_.data()[w * 64 + lane] >> b) & 1) << lane;
                    }
                    const auto rb = bits_.data()[b * words_ + w];
                    node(b)[w] =
                        rb ^ (party_ == 0 ? (shape_.equality ? (c ^ low_mask64(take)) : c) : 0);
                    node(shape_.k + b)[w] = rb & c;
                }
            }
            if (shape_.layers.empty()) {
                finish();
            }
        }

        class Opening : public Step {
            EdaBatch &o_;
            RingSlot slot_;
            bool bound_ = false, used_ = false, done_ = false;

        public:
            explicit Opening(EdaBatch &o) : o_(o) {
            }

            void bind(RoundPlan &p) override {
                if (bound_) {
                    throw std::logic_error("eda already bound");
                }
                bound_ = true;
                slot_ = p.ring(o_.shape_.k, o_.n_);
            }

            void contribute(OpenWriter &w) override {
                if (!bound_ || used_) {
                    throw std::logic_error("eda r reused");
                }
                used_ = true;
                const auto mask = native_mask<T>(o_.shape_.k);
#pragma omp parallel for if (o_.n_ >= 16384) schedule(static)
                for (std::size_t i = 0; i < o_.n_; ++i) {
                    o_.send_.data()[i] = (o_.x_[i] + o_.r_.data()[i]) & mask;
                }
                w.ring<T>(slot_, o_.send_.span());
            }

            void consume(OpenReader &r) override {
                if (!used_ || done_) {
                    throw std::logic_error("eda consume state");
                }
                done_ = true;
                r.ring<T>(slot_, o_.hat_.span());
                o_.initialize();
            }
        };

        class Layer : public Step {
            EdaBatch &o_;
            std::size_t level_, start_;
            AlignedBuffer<std::uint64_t> opened_;
            std::vector<BitSlot> slots_;
            bool bound_ = false, used_ = false, done_ = false;

        public:
            Layer(EdaBatch &o, std::size_t level, std::size_t start)
                : o_(o), level_(level), start_(start),
                  opened_(2 * o.shape_.layers[level].size() * o.words_),
                  slots_(2 * o.shape_.layers[level].size()) {
            }

            void bind(RoundPlan &p) override {
                if (bound_) {
                    throw std::logic_error("eda layer bound twice");
                }
                bound_ = true;
                for (auto &s : slots_) {
                    s = p.bits(o_.n_);
                }
            }

            void contribute(OpenWriter &w) override {
                if (!bound_ || used_) {
                    throw std::logic_error("eda triples reused");
                }
                used_ = true;
                const auto &gates = o_.shape_.layers[level_];
#pragma omp parallel for if (o_.n_ >= 16384) schedule(static)
                for (std::size_t j = 0; j < gates.size(); ++j) {
                    for (std::size_t q = 0; q < o_.words_; ++q) {
                        opened_.data()[(2 * j) * o_.words_ + q] =
                            o_.node(gates[j].a)[q] ^
                            o_.triples_.data()[(3 * (start_ + j)) * o_.words_ + q];
                        opened_.data()[(2 * j + 1) * o_.words_ + q] =
                            o_.node(gates[j].b)[q] ^
                            o_.triples_.data()[(3 * (start_ + j) + 1) * o_.words_ + q];
                    }
                }
                for (std::size_t j = 0; j < slots_.size(); ++j) {
                    w.bits(slots_[j], opened_.span().subspan(j * o_.words_, o_.words_));
                }
            }

            void consume(OpenReader &r) override {
                if (!used_ || done_) {
                    throw std::logic_error("eda layer consume state");
                }
                done_ = true;
                for (std::size_t j = 0; j < slots_.size(); ++j) {
                    r.bits(slots_[j], opened_.span().subspan(j * o_.words_, o_.words_));
                }
                const auto &gates = o_.shape_.layers[level_];
#pragma omp parallel for if (o_.n_ >= 16384) schedule(static)
                for (std::size_t j = 0; j < gates.size(); ++j) {
                    for (std::size_t q = 0; q < o_.words_; ++q) {
                        const auto d = opened_.data()[2 * j * o_.words_ + q],
                                   e = opened_.data()[(2 * j + 1) * o_.words_ + q];
                        auto value =
                            o_.triples_.data()[(3 * (start_ + j) + 2) * o_.words_ + q] ^
                            (d & o_.triples_.data()[(3 * (start_ + j) + 1) * o_.words_ + q]) ^
                            (e & o_.triples_.data()[3 * (start_ + j) * o_.words_ + q]) ^
                            (o_.party_ == 0 ? (d & e) : 0);
                        if (gates[j].xor_node >= 0) {
                            value ^= o_.node(static_cast<unsigned>(gates[j].xor_node))[q];
                        }
                        o_.node(gates[j].out)[q] = value;
                    }
                }
                if (level_ + 1 == o_.shape_.layers.size()) {
                    o_.finish();
                }
            }
        };

        std::vector<std::unique_ptr<Step>> stages_;

    public:
        EdaBatch(MaterialStore &s,
                 const MaterialPlan &p,
                 std::size_t c,
                 std::span<const T> x,
                 std::span<std::uint64_t> out)
            : shape_(p.comparisons.at(c).width, p.comparisons[c].kind == CompareKind::EdaEq),
              party_(s.party()), n_(p.comparisons[c].count), words_(bits_words(n_)), x_(x),
              output_(out), r_(n_), hat_(n_), send_(n_), bits_(shape_.k * words_),
              nodes_(shape_.nodes * words_), triples_(3 * shape_.gates * words_) {
            const auto kind = p.comparisons[c].kind;
            if ((kind != CompareKind::EdaMsb && kind != CompareKind::EdaEq) || x.size() != n_ ||
                out.size() != words_ || shape_.k > sizeof(T) * 8) {
                throw std::invalid_argument("eda batch kind/shape");
            }
            s.claim_comparison(c, 0);
            const auto &ids = s.layout().comparisons[c];
            unpack_native<T>(shape_.k, s.packed(ids.aux[0]), r_.span());
            const auto binary = s.packed(ids.aux[1]), triples = s.packed(ids.evaluations[0][0]);
            for (unsigned b = 0; b < shape_.k; ++b) {
                load_bits(binary, b * n_, n_, bits_.span().subspan(b * words_, words_));
            }
            for (std::size_t t = 0; t < 3 * shape_.gates; ++t) {
                load_bits(triples, t * n_, n_, triples_.span().subspan(t * words_, words_));
            }
            stages_.push_back(std::make_unique<Opening>(*this));
            std::size_t offset = 0;
            for (std::size_t l = 0; l < shape_.layers.size(); ++l) {
                stages_.push_back(std::make_unique<Layer>(*this, l, offset));
                offset += shape_.layers[l].size();
            }
        }

        std::size_t stage_count() const noexcept {
            return stages_.size();
        }

        Step &stage(std::size_t i) {
            return *stages_.at(i);
        }
    };
}
