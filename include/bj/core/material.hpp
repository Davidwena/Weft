#pragma once
#include "bj/common/aligned.hpp"
#include "bj/common/prg.hpp"
#include "bj/common/packed.hpp"
#include <array>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>

namespace bj {
    enum class MaterialMode : unsigned {
        Random = 0,
        Prg = 1
    };
    MaterialMode parse_material_mode(const std::string &text);
    const char *material_mode_name(MaterialMode mode) noexcept;

    struct InputSpec {
        std::string name;
        unsigned owner = 0, width = 0;
        std::size_t count = 0;
    };

    struct GateSpec {
        std::string name;
        unsigned m = 1, width = 32;
        std::size_t count = 0;
        std::vector<std::size_t> inputs;
    };
    enum class CompareKind : unsigned {
        LTBits,
        MsbKnown,
        MsbFresh,
        Eq,
        LtsKnown,
        LtsFresh,
        EdaMsb,
        EdaEq,
        Interval,
        IntervalFresh,
        EqFresh
    };
    CompareKind parse_compare_kind(const std::string &text);
    const char *compare_kind_name(CompareKind kind) noexcept;

    struct CompareSpec {
        std::string name;
        CompareKind kind = CompareKind::MsbKnown;
        unsigned width = 32, d = 0, evaluations = 1;
        std::vector<unsigned> evaluation_sources;
        std::string table_source;
        std::size_t count = 0;
        std::vector<std::size_t> inputs;
        unsigned block_width() const;
    };

    struct ComparisonSegments {
        std::vector<std::size_t> tables, aux;
        std::vector<std::vector<std::size_t>> evaluations;
    };

    struct MaskTerm {
        std::size_t source = 0, source_offset = 0, destination_offset = 0, length = 0, divisor = 1,
                    modulus = 1;
        std::int64_t coefficient = 1;
    };

    struct MaskSpec {
        std::string name;
        unsigned width = 0;
        std::size_t count = 0;
        bool fresh = true, distribute = true;
        std::vector<MaskTerm> terms;
    };

    struct ShuffleSpec {
        std::string name;
        std::size_t count = 0;
        std::vector<std::size_t> inputs, outputs;
    };

    struct ShuffleSegments {
        std::vector<std::size_t> permutations;
        std::vector<std::vector<std::size_t>> corrections;
    };

    struct BitMulSpec {
        std::string name;
        unsigned width = 64;
        std::size_t count = 0, mask = 0;
    };

    struct MultiplySpec {
        std::string name;
        unsigned width = 64;
        std::size_t count = 0;
    };

    struct MaterialPlan {
        unsigned n = 3, king = 0, output_party = 0;
        std::string open_mode = "auto", output_mode = "values";
        std::vector<InputSpec> inputs;
        std::vector<BitMulSpec> bitmuls;
        std::vector<GateSpec> gates;
        std::vector<CompareSpec> comparisons;
        std::vector<MultiplySpec> multiplications;
        std::vector<MaskSpec> masks;
        std::vector<ShuffleSpec> shuffles;
        std::string context;
        static MaterialPlan load(const std::filesystem::path &path);
        void validate() const;
        std::string canonical() const;
        std::uint64_t fingerprint() const;
    };
    enum class SegmentRole : unsigned {
        InputShare = 0,
        OwnerMask = 1,
        GateMask = 2,
        GateProduct = 3,
        CompareTable = 4,
        CompareEval = 5,
        CompareAux = 6,
        MaskShare = 7,
        ShufflePrivate = 8,
        Multiply = 9,
        BitMul = 10
    };
    enum class SegmentStorage : unsigned {
        Absent = 0,
        Dense = 1,
        Seeded = 2
    };

    struct Segment {
        std::size_t id = 0, index = 0, subset = 0, count = 0, offset = 0, bytes = 0;
        unsigned width = 0;
        SegmentRole role = SegmentRole::InputShare;
        SegmentStorage storage = SegmentStorage::Absent;
        std::uint64_t bits = 0, name_hash = 0;
    };

    struct MaterialLayout {
        std::vector<Segment> segments;
        std::vector<std::size_t> input_shares, owner_masks, gate_masks;
        std::vector<std::vector<std::size_t>> gate_products;
        std::vector<ComparisonSegments> comparisons;
        std::vector<std::array<std::size_t, 3>> multiplications, bitmuls;
        std::vector<std::size_t> masks;
        std::vector<ShuffleSegments> shuffles;
        std::size_t file_bytes = 0;
        std::uint64_t logical_share_bits = 0, gate_bits = 0, dense_share_bytes = 0, owner_bytes = 0,
                      seed_bytes = 0;
        MaterialLayout(const MaterialPlan &plan, unsigned party, MaterialMode mode);
    };

    struct DealerReport {
        double seconds = 0;
        std::array<std::byte, 16> run_id{};
        std::vector<std::uint64_t> file_bytes, share_bits, gate_bits, dense_share_bytes,
            owner_bytes;
    };

    DealerReport generate_materials(const MaterialPlan &plan,
                                    MaterialMode mode,
                                    const std::filesystem::path &directory);
    void install_dealer_network_guard();

    class MaterialStore {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        MaterialStore(const MaterialPlan &plan, unsigned party, const std::filesystem::path &file);
        ~MaterialStore();
        MaterialStore(const MaterialStore &) = delete;
        MaterialStore &operator=(const MaterialStore &) = delete;
        const MaterialLayout &layout() const noexcept;
        MaterialMode mode() const noexcept;
        unsigned party() const noexcept;
        std::span<const std::byte, 16> run_id() const noexcept;
        void preload();

        void release_decoded_segments(std::span<const std::size_t>);
        void claim_comparison(std::size_t comparison, unsigned evaluation);
        void claim_input(std::size_t input);
        void claim_multiply(std::size_t index);
        void claim_bitmul(std::size_t index);
        void claim_shuffle(std::size_t shuffle);
        void claim_mask(std::size_t mask);
        void claim_gate(std::size_t gate);
        void claim_execution();
        void begin_online();
        std::span<const std::byte> packed(std::size_t segment) const;
    };
}
