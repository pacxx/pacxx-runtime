//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_CUDARUNTIME_H
#define PACXX_V2_CUDARUNTIME_H
#include <string>
#include <memory>
#include <map>
#include <list>
#include "detail/IRRuntime.h"
#include "detail/cuda/CUDAKernel.h"
#include "detail/cuda/CUDADeviceBuffer.h"
// forward declarations of cuda driver structs
struct CUctx_st;
typedef struct CUctx_st* CUcontext;
struct CUmod_st;
typedef struct CUmod_st* CUmodule;



namespace pacxx
{
  namespace v2
  {
    class CUDARuntime : public IRRuntime
    {
    public:
      CUDARuntime(unsigned dev_id);
      virtual ~CUDARuntime();

      virtual void linkMC(const std::string& MC) override;
      virtual Kernel& getKernel(const std::string& name) override;

      virtual size_t getPreferedMemoryAlignment() override;
      virtual DeviceBufferBase* allocateMemory(size_t bytes) override;
      virtual RawDeviceBuffer* allocateRawMemory(size_t bytes) override;
      virtual void deleteRawMemory(RawDeviceBuffer* ptr) override;

    private:
      CUcontext _context;
      CUmodule _mod;
      std::map<std::string, std::unique_ptr<CUDAKernel>> _kernels;
      std::list<std::unique_ptr<DeviceBufferBase>> _memory;
    };
  }
}

#endif //PACXX_V2_CUDARUNTIME_H
