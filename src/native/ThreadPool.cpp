//
// Created by lars on 09/11/16.
//

#include "detail/native/ThreadPool.h"

#include "detail/common/Log.h"

namespace pacxx {
    namespace v2 {

        ThreadPool::ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

        ThreadPool::ThreadPool(unsigned ThreadCount) : ActiveThreads(0), EnableFlag(true) {

            __verbose("ThreadPool create with: ", ThreadCount, " threads");

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

    }
}
