#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

template <typename Input, typename Output>
class OrderedParallelProcessor final {
public:
    using Processor = std::function<Output(Input)>;

    OrderedParallelProcessor(std::size_t num_threads, Processor processor)
        : processor_(std::move(processor)) {
        if (num_threads == 0)
            num_threads = 1;
        workers_.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; i++)
            workers_.emplace_back(&OrderedParallelProcessor::workerLoop, this);
    }

    OrderedParallelProcessor(const OrderedParallelProcessor &) = delete;
    OrderedParallelProcessor &operator=(const OrderedParallelProcessor &) = delete;

    ~OrderedParallelProcessor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        jobs_ready_.notify_all();
        results_ready_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable())
                worker.join();
        }
    }

    void submit(std::size_t index, Input input) {
        std::lock_guard<std::mutex> lock(mutex_);
        rethrowFailureLocked();

        jobs_.push_back(Job{index, std::move(input)});
        jobs_ready_.notify_one();
    }

    bool tryTakeNext(std::size_t index, Output &output) {
        std::lock_guard<std::mutex> lock(mutex_);
        rethrowFailureLocked();

        auto it = results_.find(index);
        if (it == results_.end())
            return false;

        output = std::move(it->second);
        results_.erase(it);
        return true;
    }

    Output takeNext(std::size_t index) {
        std::unique_lock<std::mutex> lock(mutex_);
        results_ready_.wait(lock, [&]() {
            return failure_ || results_.find(index) != results_.end();
        });
        rethrowFailureLocked();

        auto it = results_.find(index);
        Output output = std::move(it->second);
        results_.erase(it);
        return output;
    }

    void closeInput() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            input_closed_ = true;
        }
        jobs_ready_.notify_all();
    }

private:
    struct Job {
        std::size_t index;
        Input input;
    };

    void workerLoop() {
        while (true) {
            Job job{0, Input{}};
            {
                std::unique_lock<std::mutex> lock(mutex_);
                jobs_ready_.wait(lock, [&]() {
                    return stopping_ || failure_ || !jobs_.empty() || input_closed_;
                });

                if (stopping_ || failure_)
                    return;
                if (jobs_.empty()) {
                    if (input_closed_)
                        return;
                    continue;
                }

                job = std::move(jobs_.front());
                jobs_.pop_front();
            }

            try {
                Output output = processor_(std::move(job.input));
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    results_.emplace(job.index, std::move(output));
                }
                results_ready_.notify_all();
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!failure_)
                        failure_ = std::current_exception();
                }
                jobs_ready_.notify_all();
                results_ready_.notify_all();
                return;
            }
        }
    }

    void rethrowFailureLocked() const {
        if (failure_)
            std::rethrow_exception(failure_);
    }

    Processor processor_;
    std::vector<std::thread> workers_;

    mutable std::mutex mutex_;
    std::condition_variable jobs_ready_;
    std::condition_variable results_ready_;
    std::deque<Job> jobs_;
    std::map<std::size_t, Output> results_;
    bool input_closed_ = false;
    bool stopping_ = false;
    std::exception_ptr failure_;
};
