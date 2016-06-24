//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_IRRUNTIME_H
#define PACXX_V2_IRRUNTIME_H

#include <string>
#include <vector>
#include <llvm/IR/Module.h>
#include "Kernel.h"
#include "DeviceBuffer.h"

namespace pacxx
{
  namespace v2
  {
    class IRRuntimeBase
    {
    public:
      virtual ~IRRuntimeBase() {};
      virtual void link(std::unique_ptr<llvm::Module> M) = 0;
      virtual Kernel& getKernel(const std::string& name) = 0;

      virtual size_t getPreferedMemoryAlignment() = 0;

      virtual RawDeviceBuffer* allocateRawMemory(size_t bytes) = 0;
      virtual void deleteRawMemory(RawDeviceBuffer* ptr) = 0;

      virtual const llvm::Module& getModule() = 0;

      virtual void synchronize() = 0;
    };

    template <typename Derived> // CRTP
    class IRRuntime : public IRRuntimeBase
    {
    private:
      auto& derived() { return *static_cast<Derived*>(this); }
    public:
      virtual ~IRRuntime() {};
      virtual void link(std::unique_ptr<llvm::Module> M) override
      {
        derived().link(std::move(M));
      }
      virtual Kernel& getKernel(const std::string& name) override
      {
        return derived().getKernel(name);
      }

      virtual size_t getPreferedMemoryAlignment() override
      {
        return derived().getPreferedMemoryAlignment();
      }

      template <typename T>
      DeviceBuffer<T>* allocateMemory(size_t count)
      {
        return derived().template allocateMemory<T>(count);
      }

      virtual RawDeviceBuffer* allocateRawMemory(size_t bytes) override
      {
        return derived().allocateRawMemory(bytes);
      }
      virtual void deleteRawMemory(RawDeviceBuffer* ptr) override
      {
        derived().deleteRawMemory(ptr);
      }
    };
  }
}



#endif //PACXX_V2_IRRUNTIME_H
