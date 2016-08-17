//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_IRRUNTIME_H
#define PACXX_V2_IRRUNTIME_H

#include <string>
#include <vector>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include "Kernel.h"
#include "DeviceBuffer.h"

namespace pacxx {
  namespace v2 {
    template<typename T>
    class Callback;

    class CallbackBase;

    class IRRuntimeBase {
    public:
      virtual ~IRRuntimeBase() { };

      virtual void link(std::unique_ptr<llvm::Module> M) = 0;

      virtual Kernel &getKernel(const std::string &name) = 0;

      virtual size_t getPreferedMemoryAlignment() = 0;

      virtual RawDeviceBuffer *allocateRawMemory(size_t bytes) = 0;

      virtual void deleteRawMemory(RawDeviceBuffer *ptr) = 0;

      virtual void initializeMSP(std::unique_ptr<llvm::Module> M) = 0;

      virtual void evaluateStagedFunctions(Kernel& K) = 0;

      virtual void requestIRTransformation(Kernel &K) = 0;

      virtual const llvm::Module &getModule() = 0;

      virtual void synchronize() = 0;

      virtual llvm::legacy::PassManager& getPassManager() = 0;

//      virtual void removeCallback(CallbackBase* ptr) = 0;

    };

    template<typename Derived> // CRTP
    class IRRuntime : public IRRuntimeBase {
    private:
      auto &derived() { return *static_cast<Derived *>(this); }

    public:
      template<typename T>
      DeviceBuffer<T> *allocateMemory(size_t count) {
        return derived().template allocateMemory<T>(count);
      }

      template<typename T>
      void freeMemory(DeviceBuffer<T>* ptr) {
        return derived().template freeMemory<T>(ptr);
      }

//      template<typename CallbackFunc>
//      void setCallback(Callback<CallbackFunc> cb) {
//          return derived().template setCallback<CallbackFunc>(cb);
//      }

    };
  }
}


#endif //PACXX_V2_IRRUNTIME_H
