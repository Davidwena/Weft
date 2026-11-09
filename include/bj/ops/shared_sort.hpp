#pragma once
#include "bj/core/table.hpp"
#include "bj/ops/quicksort.hpp"
#include <functional>

namespace bj {
    struct SharedSortHandle {
        std::size_t job = SIZE_MAX;
        SortStats stats{};
        double seconds = 0;
    };

    using SortBinder =
        std::function<SharedSortHandle *(const std::string &, MaskedTable &, MaskedTable &)>;
    SortBinder *sort_binder() noexcept;
    void sort_binder(SortBinder *) noexcept;
}
