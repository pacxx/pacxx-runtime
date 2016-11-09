#ifndef THREAD_POOL_H
#define THREAD_POOL_H

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

        ThreadPool::ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

        ThreadPool::ThreadPool(unsigned ThreadCount) : ActiveThreads(0), EnableFlag(true) {

          Threads.reserve(ThreadCount);
          for (unsigned ThreadID = 0; ThreadID < ThreadCount; ++ThreadID) {
            Threads.emplace_back([&] {
                while (true) {
                  PackagedTaskTy Task;
                  {
                    std::unique_lock<std::mutex> LockGuard(QueueLock);
                    QueueCondition.wait(LockGuard,
                                        [&] { return !EnableFlag || !Tasks.empty(); });
                    if (!EnableFlag && Tasks.empty())
                      return;
                    {
                      ++ActiveThreads;
                      std::unique_lock<std::mutex> LockGuard(CompletionLock);
                    }
                    Task = std::move(Tasks.front());
                    Tasks.pop();
                  }

                  Task();

                  {
                    std::unique_lock<std::mutex> LockGuard(CompletionLock);
                    --ActiveThreads;
                  }

                  CompletionCondition.notify_all();
                }
            });
          }
        }

        void ThreadPool::wait() {
          std::unique_lock<std::mutex> LockGuard(CompletionLock);
          CompletionCondition.wait(LockGuard, [&] { return !ActiveThreads && Tasks.empty(); });
        }

        std::shared_future<ThreadPool::VoidTy> ThreadPool::asyncImpl(TaskTy Task) {
          PackagedTaskTy PackagedTask(std::move(Task));
          auto Future = PackagedTask.get_future();
          {
            std::unique_lock<std::mutex> LockGuard(QueueLock);
            Tasks.push(std::move(PackagedTask));
          }
          QueueCondition.notify_one();
          return Future.share();
        }

        ThreadPool::~ThreadPool() {
          {
            std::unique_lock<std::mutex> LockGuard(QueueLock);
            EnableFlag = false;
          }
          QueueCondition.notify_all();
          for (auto &Worker : Threads)
            Worker.join();
        }

    } // end of v2 namespace
} // end of pacxx namespace
#endif
