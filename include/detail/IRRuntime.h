//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_IRRUNTIME_H
#define PACXX_V2_IRRUNTIME_H

#include <string>
#include <vector>
#include "Kernel.h"
#include "DeviceBuffer.h"

namespace pacxx
{
  namespace v2
  {
    class IRRuntime
    {
    public:
      virtual ~IRRuntime() {};
      virtual void linkMC(const std::string& MC) = 0;
      virtual Kernel& getKernel(const std::string& name) = 0;

      virtual size_t getPreferedMemoryAlignment() = 0;
      virtual DeviceBufferBase* allocateMemory(size_t bytes) = 0;
      virtual RawDeviceBuffer* allocateRawMemory(size_t bytes) = 0;
      virtual void deleteRawMemory(RawDeviceBuffer* ptr) = 0;
    };
  }
}



#endif //PACXX_V2_IRRUNTIME_H
