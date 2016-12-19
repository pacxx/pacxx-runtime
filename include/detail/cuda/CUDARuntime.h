//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_CUDARUNTIME_H
#define PACXX_V2_CUDARUNTIME_H

#include <string>
#include <memory>
#include <map>
#include <list>
#include <detail/common/Exceptions.h>
#include "../IRRuntime.h"
#include "CUDAKernel.h"
#include "CUDADeviceBuffer.h"
#include "PTXBackend.h"
#include "../msp/MSPEngine.h"
#include <cstdlib>

// forward declarations of cuda driver structs
struct CUctx_st;
typedef struct CUctx_st* CUcontext;
struct CUmod_st;
typedef struct CUmod_st* CUmodule;

namespace pacxx {
  namespace v2 {

    class CUDARuntime : public IRRuntime<CUDARuntime> {
    public:

      using CompilerT = PTXBackend;

      CUDARuntime(unsigned dev_id);

      virtual ~CUDARuntime();

      virtual void link(std::unique_ptr <llvm::Module> M) override;

      virtual Kernel& getKernel(const std::string& name) override;

      virtual size_t getPreferedMemoryAlignment() override;

      template<typename T>
      DeviceBuffer<T>* allocateMemory(size_t count, T *host_ptr) {
        if(host_ptr)
            throw common::generic_exception("using host buffer with gpu is not allowed");
        CUDARawDeviceBuffer raw;
        raw.allocate(count * sizeof(T));
        auto wrapped = new CUDADeviceBuffer<T>(std::move(raw));
        _memory.push_back(std::unique_ptr<DeviceBufferBase>(
            static_cast<DeviceBufferBase*>(wrapped)));
        return wrapped;
      }

      template<typename T>
      DeviceBuffer<T>* translateMemory(T* ptr) {
        auto It = std::find_if(_memory.begin(), _memory.end(), [&](const auto& element) {
          return reinterpret_cast<CUDADeviceBuffer<T>*>(element.get())->get() == ptr;
        });

        if (It != _memory.end())
          return reinterpret_cast<DeviceBuffer<T>*>(It->get());
        else
          throw common::generic_exception("supplied pointer not found in translation list");
      }

      template<typename T>
      void deleteMemory(DeviceBuffer<T>* ptr) {
        auto It = std::find_if(_memory.begin(), _memory.end(), [&](const auto& element) {
          return element.get() == ptr;
        });

        if (It != _memory.end())
          _memory.erase(It);
      }

      virtual RawDeviceBuffer* allocateRawMemory(size_t bytes) override;

      virtual void deleteRawMemory(RawDeviceBuffer* ptr) override;

      virtual void initializeMSP(std::unique_ptr <llvm::Module> M) override;

      virtual void evaluateStagedFunctions(Kernel& K) override;

      virtual void requestIRTransformation(Kernel& K) override;

      virtual const llvm::Module& getModule() override;

      virtual void synchronize() override;

      virtual llvm::legacy::PassManager& getPassManager() override;

//      template<typename T>
//      void setCallback(Callback<T> cb) {
//        //callbacks must survive until they are fired
//        callback_mem new_cb;
//        new_cb.size = sizeof(Callback<T>);
//        new_cb.ptr = std::malloc(new_cb.size);
//        Callback<T>* survivingCopy = new(new_cb.ptr) Callback<T>(cb);
//        survivingCopy->registeredWith(this);
//        _callbacks.push_back(new_cb);
//        SEC_CUDA_CALL(cudaStreamAddCallback(nullptr, CUDARuntime::fireCallback, new_cb.ptr, NULL));
//
//      };
//
//      virtual void removeCallback(CallbackBase* ptr) override {
//        auto It = std::find_if(_callbacks.begin(), _callbacks.end(), [&](const auto& v) { return v.ptr == ptr; });
//        if (It != _callbacks.end()) {
//          ptr->~CallbackBase();
//          std::free(It->ptr);
//          _callbacks.erase(It);
//        }
//      }

    private:
      void compileAndLink();

    public:
      static void fireCallback(cudaStream_t stream, cudaError_t status, void* userData) {
        (*reinterpret_cast<std::function<void()>*>(userData))();
      }

    private:
      CUcontext _context;
      CUmodule _mod;
      std::unique_ptr <CompilerT> _compiler;
      std::unique_ptr <llvm::Module> _M, _rawM;
      std::map <std::string, std::unique_ptr<CUDAKernel>> _kernels;
      std::list <std::unique_ptr<DeviceBufferBase>> _memory;

      struct callback_mem {
        size_t size;
        void* ptr;
      };

      std::list <callback_mem> _callbacks;
      bool _delayed_compilation;
      v2::MSPEngine _msp_engine;
    };
  }
}

#endif //PACXX_V2_CUDARUNTIME_H
