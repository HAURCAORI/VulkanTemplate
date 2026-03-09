#pragma once

// Cross-platform thread pool built on C++17 standard library only.
// No OS-specific headers ??works identically on Windows and Linux.
//
// Usage:
//   vkt::ThreadPool pool;                        // hardware_concurrency threads
//   auto f = pool.submit([](int x){ return x*2; }, 21);
//   std::cout << f.get();                        // prints 42
//   pool.waitAll();                              // block until all tasks done

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <cassert>
#include <unordered_set>

namespace vkt {

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency()) {
        if (numThreads == 0) numThreads = 1;
        m_workers.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i) {
            m_workers.emplace_back([this] { workerLoop(); });
            m_workerIds.insert(m_workers.back().get_id());
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (auto& w : m_workers) {
            if (w.joinable()) w.join();
        }
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Submit a callable with arguments. Returns a std::future for the result.
    // Throws std::runtime_error if the pool has been stopped.
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using Ret = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<Ret()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<Ret> future = task->get_future();

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_stop)
                throw std::runtime_error("[ThreadPool] Cannot submit to a stopped pool");
            ++m_activeTasks;
            m_tasks.emplace([task]() { (*task)(); });
        }
        m_cv.notify_one();
        return future;
    }

    // Block the calling thread until all submitted tasks have completed.
    // MUST NOT be called from a worker thread: doing so causes a self-deadlock
    // because the worker holds an active-task slot that can never reach zero.
    void waitAll() {
        // Checked in both debug and release builds: the assert only fires in debug,
        // but the throw protects release builds from a silent permanent hang.
        if (m_workerIds.count(std::this_thread::get_id()))
            throw std::logic_error(
                "[ThreadPool] waitAll() called from a worker thread (self-deadlock)");
        std::unique_lock<std::mutex> lock(m_mutex);
        m_doneCv.wait(lock, [this] { return m_activeTasks == 0; });
    }

    size_t threadCount() const { return m_workers.size(); }

    size_t pendingCount() const {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_tasks.size();
    }

private:
    void workerLoop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this] { return m_stop || !m_tasks.empty(); });
                if (m_stop && m_tasks.empty()) return;
                task = std::move(m_tasks.front());
                m_tasks.pop();
            }
            // Execute the task outside the lock so other threads can pick up work
            task();
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                --m_activeTasks;
                m_doneCv.notify_all();
            }
        }
    }

    std::vector<std::thread>                  m_workers;
    std::unordered_set<std::thread::id>       m_workerIds;  // for self-deadlock detection
    std::queue<std::function<void()>>         m_tasks;
    mutable std::mutex                        m_mutex;
    std::condition_variable                   m_cv;         // wakes workers when tasks arrive
    std::condition_variable                   m_doneCv;     // wakes waitAll() callers
    size_t                                    m_activeTasks{0}; // queued + running
    bool                                      m_stop{false};
};

} // namespace vkt
