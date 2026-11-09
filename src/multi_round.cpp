#include "bj/core/multi_round.hpp"
#include <condition_variable>
#include <exception>
#include <mutex>
#include <omp.h>
#include <thread>

namespace bj {
    struct MultiRound::Impl {
        enum State {
            Dormant,
            Runnable,
            Running,
            Open,
            Shuffling,
            Waiting,
            Done
        };

        struct Worker final : NetworkHook {
            Impl &owner;
            std::size_t id;
            State state = Dormant;
            bool ready = false;
            OpenBuf *buf = nullptr;
            std::span<Shuffle *const> shuffles;
            std::size_t target = SIZE_MAX;
            std::exception_ptr error;
            std::thread thread;

            Worker(Impl &o, std::size_t i) : owner(o), id(i) {
            }

            void yield(State next) {
                std::unique_lock lock(owner.mutex);
                state = next;
                owner.cv.notify_all();
                owner.cv.wait(lock, [&] {
                    return owner.stop || state == Running;
                });
                if (owner.stop) {
                    throw std::runtime_error("multi execution cancelled");
                }
            }

            void open(OpenBuf &b) override {
                buf = &b;
                yield(Open);
            }

            void shuffle(std::span<Shuffle *const> s) override {
                shuffles = s;
                yield(Shuffling);
            }

            void wait_shared(std::size_t target_job) override {
                target = target_job;
                yield(Waiting);
            }

            void body() noexcept {
                std::exception_ptr failure;
                try {
                    omp_set_dynamic(0);
                    omp_set_num_threads(static_cast<int>(owner.team_threads));
#pragma omp parallel
                    { asm volatile("" ::: "memory"); }
                    {
                        std::unique_lock lock(owner.mutex);
                        ready = true;
                        owner.cv.notify_all();
                        owner.cv.wait(lock, [&] {
                            return owner.stop || state == Running;
                        });
                        if (owner.stop) {
                            return;
                        }
                    }
                    Net::thread_hook(this);
                    owner.jobs[id].run();
                    Net::thread_hook(nullptr);
                } catch (...) {
                    Net::thread_hook(nullptr);
                    failure = std::current_exception();
                }
                {
                    std::lock_guard lock(owner.mutex);
                    error = failure;
                    state = Done;
                }
                owner.cv.notify_all();
            }
        };

        Net &net;
        std::vector<ScheduledJob> jobs;
        std::vector<std::unique_ptr<Worker>> workers;
        std::mutex mutex;
        std::condition_variable cv;
        bool stop = false, used = false;
        unsigned threads, team_threads, max_active;
        bool merge;
        OpenLayout layout;
        AlignedBuffer<std::byte> tx, rx;
        ShuffleBatch shuffle;
        std::vector<OpenField> fields;
        std::vector<Worker *> opens, chains;
        std::vector<Shuffle *> group;
        MergeStats stats;
        OpenAugment *augment;

        Impl(Net &n,
             std::vector<ScheduledJob> j,
             std::size_t bytes,
             std::size_t nf,
             unsigned t,
             bool m,
             OpenAugment *a)
            : net(n), jobs(std::move(j)), threads(t),
              team_threads(std::max(
                  1u,
                  t / std::max(1u, static_cast<unsigned>(std::min<std::size_t>(t, jobs.size()))))),
              max_active(static_cast<unsigned>(std::min<std::size_t>(t, jobs.size()))), merge(m),
              tx(bytes), rx(bytes), shuffle(bytes), augment(a) {
            if (jobs.empty() || !t || !nf || !bytes) {
                throw std::invalid_argument("multi scheduler dimensions");
            }
            layout.reserve_batch(nf, bytes);
            fields.reserve(nf);
            opens.reserve(jobs.size());
            chains.reserve(jobs.size());
            group.reserve(2 * jobs.size());
            stats.wave_rounds.reserve(bytes / 16 + 1024);
            for (std::size_t i = 0; i < jobs.size(); ++i) {
                for (auto dep : jobs[i].dependencies) {
                    if (dep >= jobs.size() || dep == i) {
                        throw std::invalid_argument("invalid shared job dependency");
                    }
                }
                workers.push_back(std::make_unique<Worker>(*this, i));
            }
            try {
                for (auto &w : workers) {
                    w->thread = std::thread([ptr = w.get()] {
                        ptr->body();
                    });
                }
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] {
                    return std::all_of(workers.begin(), workers.end(), [](auto &w) {
                        return w->ready || w->error;
                    });
                });
                for (auto &w : workers) {
                    if (w->error) {
                        std::rethrow_exception(w->error);
                    }
                }
            } catch (...) {
                shutdown();
                throw;
            }
        }

        void shutdown() noexcept {
            {
                std::lock_guard lock(mutex);
                stop = true;
            }
            cv.notify_all();
            for (auto &w : workers) {
                if (w->thread.joinable()) {
                    w->thread.join();
                }
            }
        }

        ~Impl() {
            shutdown();
        }

        void activate_shared() {
            for (std::size_t i = 0; i < workers.size(); ++i) {
                auto &w = *workers[i];
                if (w.state != Dormant || jobs[i].dependencies.empty()) {
                    continue;
                }
                if (std::all_of(
                        jobs[i].dependencies.begin(), jobs[i].dependencies.end(), [&](auto dep) {
                            return workers[dep]->state == Waiting && workers[dep]->target == i;
                        })) {
                    w.state = Runnable;
                }
            }
            for (auto &w : workers) {
                if (w->state == Waiting && workers.at(w->target)->state == Done) {
                    w->state = Runnable;
                }
            }
        }

        void open_group(std::span<Worker *const> batch) {
            fields.clear();
            std::size_t bytes = 0;
            for (auto *w : batch) {
                const auto &b = *w->buf;
                if (b.layout.bytes() > tx.size() - bytes) {
                    throw std::length_error("merged open byte bound");
                }
                for (auto f : b.layout.fields()) {
                    if (fields.size() == fields.capacity()) {
                        throw std::length_error("merged open field bound");
                    }
                    fields.push_back(f);
                }
                std::copy(b.share.begin(), b.share.end(), tx.data() + bytes);
                bytes += b.share.size();
            }
            const auto extras = augment ? augment->prepare() : std::span<OpenBuf *const>{};
            for (auto *extra : extras) {
                if (extra->share.size() > tx.size() - bytes) {
                    throw std::length_error("shared endpoint batch bound");
                }
                for (auto f : extra->layout.fields()) {
                    if (fields.size() == fields.capacity()) {
                        throw std::length_error("shared endpoint field bound");
                    }
                    fields.push_back(f);
                }
                std::copy(extra->share.begin(), extra->share.end(), tx.data() + bytes);
                bytes += extra->share.size();
            }
            layout.configure_batch(fields);
            if (layout.bytes() != bytes) {
                throw std::logic_error("merged open byte accounting");
            }
            OpenBuf b(layout, tx.span().first(bytes), rx.span().first(bytes));
            net.open_round(b);
            std::size_t off = 0;
            for (auto *w : batch) {
                std::copy_n(rx.data() + off, w->buf->opened.size(), w->buf->opened.data());
                off += w->buf->opened.size();
                ++w->counters.rounds;
                w->counters.hops += net.uses_king() ? 2 : 1;
                w->counters.contribution_bits += w->buf->layout.bits();
            }
            for (auto *extra : extras) {
                std::copy_n(rx.data() + off, extra->opened.size(), extra->opened.data());
                off += extra->opened.size();
            }
            if (augment) {
                augment->consume();
            }
            stats.unmerged_opens += batch.size();
            ++stats.merged_opens;
            stats.contribution_bits += layout.bits();
        }

        void shuffle_group(std::span<Worker *const> batch) {
            group.clear();
            for (auto *w : batch) {
                for (auto *s : w->shuffles) {
                    if (group.size() == group.capacity()) {
                        throw std::length_error("shuffle group bound");
                    }
                    group.push_back(s);
                    stats.shuffle_bytes += s->bytes();
                }
            }
            shuffle.run(net, group);
            stats.unmerged_shuffle_hops += batch.size() * net.size();
            stats.merged_shuffle_hops += net.size();
        }

        void run() {
            if (used) {
                throw std::logic_error("multi schedule already used");
            }
            used = true;
            std::unique_lock lock(mutex);
            for (std::size_t i = 0; i < workers.size(); ++i) {
                if (jobs[i].dependencies.empty()) {
                    workers[i]->state = Runnable;
                }
            }
            cv.notify_all();
            for (;;) {
                unsigned dispatched = 0;
                for (auto &w : workers) {
                    if (w->state == Runnable && dispatched < max_active) {
                        w->state = Running;
                        ++dispatched;
                    }
                }
                cv.notify_all();
                cv.wait(lock, [&] {
                    return std::none_of(workers.begin(), workers.end(), [](auto &w) {
                        return w->state == Running;
                    });
                });
                for (auto &w : workers) {
                    if (w->error) {
                        std::rethrow_exception(w->error);
                    }
                }
                if (std::any_of(workers.begin(), workers.end(), [](auto &w) {
                        return w->state == Runnable;
                    })) {
                    continue;
                }
                activate_shared();
                if (std::any_of(workers.begin(), workers.end(), [](auto &w) {
                        return w->state == Runnable;
                    })) {
                    cv.notify_all();
                    continue;
                }
                if (std::all_of(workers.begin(), workers.end(), [](auto &w) {
                        return w->state == Done;
                    })) {
                    break;
                }
                opens.clear();
                chains.clear();
                for (auto &w : workers) {
                    if (w->state == Open) {
                        opens.push_back(w.get());
                    }
                    if (w->state == Shuffling) {
                        chains.push_back(w.get());
                    }
                }
                if (opens.empty() && chains.empty()) {
                    throw std::logic_error("cyclic/unsatisfied multi dependency");
                }
                std::uint64_t rounds = 0;
                if (!opens.empty()) {
                    if (merge) {
                        open_group(opens);
                        ++rounds;
                    } else {
                        for (auto *w : opens) {
                            open_group(std::span<Worker *const>(&w, 1));
                            ++rounds;
                        }
                    }
                }
                if (!chains.empty()) {
                    if (merge) {
                        shuffle_group(chains);
                        rounds += net.size();
                    } else {
                        for (auto *w : chains) {
                            shuffle_group(std::span<Worker *const>(&w, 1));
                            rounds += net.size();
                        }
                    }
                }

                if (stats.wave_rounds.size() == stats.wave_rounds.capacity()) {
                    throw std::length_error("multi DAG trace bound");
                }
                stats.wave_rounds.push_back(rounds);
                stats.graph_rounds += rounds;
                for (auto *w : opens) {
                    w->state = Runnable;
                }
                for (auto *w : chains) {
                    w->state = Runnable;
                }
                cv.notify_all();
            }
        }
    };

    MultiRound::MultiRound(Net &n,
                           std::vector<ScheduledJob> j,
                           std::size_t bytes,
                           std::size_t fields,
                           unsigned t,
                           bool merge,
                           OpenAugment *augment)
        : impl_(std::make_unique<Impl>(n, std::move(j), bytes, fields, t, merge, augment)) {
    }

    MultiRound::~MultiRound() = default;

    void MultiRound::run() {
        impl_->run();
    }

    const MergeStats &MultiRound::stats() const {
        return impl_->stats;
    }
}
