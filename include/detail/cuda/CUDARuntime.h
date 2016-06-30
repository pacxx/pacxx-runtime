//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_CUDARUNTIME_H
#define PACXX_V2_CUDARUNTIME_H
#include <string>
#include <memory>
#include <map>
#include <list>
#include "../IRRuntime.h"
#include "CUDAKernel.h"
#include "CUDADeviceBuffer.h"
#include "PTXBackend.h"
#include "../msp/MSPEngine.h"

// forward declarations of cuda driver structs
struct CUctx_st;
typedef struct CUctx_st* CUcontext;
struct CUmod_st;
typedef struct CUmod_st* CUmodule;

namespace pacxx
{
  namespace v2
  {
    class CUDARuntime : public IRRuntime<CUDARuntime>
    {
    public:

      using CompilerT = PTXBackend;

      CUDARuntime(unsigned dev_id);
      virtual ~CUDARuntime();

      virtual void link(std::unique_ptr<llvm::Module> M) override;
      virtual Kernel& getKernel(const std::string& name) override;

      virtual size_t getPreferedMemoryAlignment() override;

      template <typename T>
      DeviceBuffer<T>* allocateMemory(size_t count) {
        CUDARawDeviceBuffer raw;
        raw.allocate(count * sizeof(T));
        auto wrapped = new CUDADeviceBuffer<T>(std::move(raw));
        _memory.push_back(std::unique_ptr<DeviceBufferBase>(
            static_cast<DeviceBufferBase *>(wrapped)));
        return wrapped;
      }

      virtual RawDeviceBuffer* allocateRawMemory(size_t bytes) override;
      virtual void deleteRawMemory(RawDeviceBuffer* ptr) override;

      virtual void initializeMSP(std::unique_ptr<llvm::Module> M) override;
      virtual void evaluateStagedFunctions(Kernel& K) override;
      virtual void requestIRTransformation(Kernel& K) override;

      virtual const llvm::Module& getModule() override;

      virtual void synchronize() override;

    private:
      void compileAndLink();

    private:
      CUcontext _context;
      CUmodule _mod;
      std::unique_ptr<CompilerT> _compiler;
      std::unique_ptr<llvm::Module> _M, _rawM;
      std::map<std::string, std::unique_ptr<CUDAKernel>> _kernels;
      std::list<std::unique_ptr<DeviceBufferBase>> _memory;
      bool _delayed_compilation;
      v2::MSPEngine _msp_engine;
    };
  }
}

#endif //PACXX_V2_CUDARUNTIME_H
