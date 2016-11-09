#ifndef PACXX_V2_THREAD_POOL_H
#define PACXX_V2_THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <utility>

//ThreadPool from llvm

namespace pacxx {
    namespace v2 {


        class ThreadPool {

        public:
            using VoidTy = void;
            using TaskTy = std::function<void()>;
            using PackagedTaskTy = std::packaged_task<void()>;

            ThreadPool();

            ThreadPool(unsigned ThreadCount);

            ~ThreadPool();

            template<typename Function, typename... Args>
            inline std::shared_future<VoidTy> async(Function &&F, Args &&... ArgList) {
              auto Task = std::bind(std::forward<Function>(F), std::forward<Args>(ArgList)...);
              return asyncImpl(std::move(Task));
            }

            template<typename Function>
            inline std::shared_future<VoidTy> async(Function &&F) {
              return asyncImpl(std::forward<Function>(F));
            }

            void wait();

        private:
            std::shared_future<VoidTy> asyncImpl(TaskTy F);

            std::vector<std::thread> Threads;

            std::queue<PackagedTaskTy> Tasks;

            std::mutex QueueLock;
            std::condition_variable QueueCondition;

            std::mutex CompletionLock;
            std::condition_variable CompletionCondition;

            std::atomic_uint ActiveThreads;

            bool EnableFlag;
        };

    } // end of v2 namespace
} // end of pacxx namespace
#endif
