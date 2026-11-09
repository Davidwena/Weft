#include "bj/ops/shared_sort.hpp"

namespace bj {
    namespace {
        thread_local SortBinder *binder = nullptr;
    }

    SortBinder *sort_binder() noexcept {
        return binder;
    }

    void sort_binder(SortBinder *value) noexcept {
        binder = value;
    }
}
