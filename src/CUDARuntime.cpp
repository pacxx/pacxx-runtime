//
// Created by mhaidl on 30/05/16.
//

#include "detail/cuda/CUDARuntime.h"
#include "detail/common/Log.h"
#include "detail/cuda/CUDAErrorDetection.h"
#include <cuda.h>
#include <detail/Kernel.h>
#include <detail/common/Exceptions.h>
#include <detail/cuda/CUDADeviceBuffer.h>
#include <string>

namespace pacxx {
namespace v2 {
CUDARuntime::CUDARuntime(unsigned dev_id) : _context(nullptr) {
  SEC_CUDA_CALL(cuInit(0));
  CUcontext old;
  SEC_CUDA_CALL(cuCtxGetCurrent(&old)); // check if there is already a context
  if (old) {
    _context = old; // we use the found context
  }
  if (!_context) { // create a new context for the device
    CUdevice device;
    SEC_CUDA_CALL(cuDeviceGet(&device, dev_id));
    SEC_CUDA_CALL(cuCtxCreate(&_context, CU_CTX_SCHED_AUTO, device));
    __verbose("creating cudaCtx for device: ", dev_id, " ", device, " ",
              _context);
  }
}

CUDARuntime::~CUDARuntime() {
  __message("dtor of cuda runtime");
}

void CUDARuntime::linkMC(const std::string &MC) {
  float walltime;
  char error_log[81920];
  char info_log[81920];
  size_t logSize = 81920;

  // Setup linker options

  CUjit_option lioptions[] = {
      CU_JIT_WALL_TIME,                  // Return walltime from JIT compilation
      CU_JIT_INFO_LOG_BUFFER,            // Pass a buffer for info messages
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, // Pass the size of the info buffer
      CU_JIT_ERROR_LOG_BUFFER,           // Pass a buffer for error message
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, // Pass the size of the error buffer
      CU_JIT_LOG_VERBOSE                  // Make the linker verbose
  };

  void *opt_values[] = {
      reinterpret_cast<void *>(&walltime), reinterpret_cast<void *>(info_log),
      reinterpret_cast<void *>(logSize),   reinterpret_cast<void *>(error_log),
      reinterpret_cast<void *>(logSize),   reinterpret_cast<void *>(1)};

  SEC_CUDA_CALL(
      cuModuleLoadDataEx(&_mod, MC.c_str(), 6, lioptions, opt_values));
  if (info_log[0] != '\0')
    __warning("Linker Output: \n", info_log);
}

Kernel &CUDARuntime::getKernel(const std::string &name) {
  auto It = std::find_if(_kernels.begin(), _kernels.end(),
                         [&](const auto &p) { return name == p.first; });
  if (It == _kernels.end()) {

    CUfunction ptr;
    SEC_CUDA_CALL(cuModuleGetFunction(&ptr, _mod, name.c_str()));
    if (!ptr)
      throw common::generic_exception("Kernel function not found in module!");
    auto kernel = new CUDAKernel(ptr);
    _kernels[name].reset(kernel);

    return *kernel;
  } else {
    return *It->second;
  }
}

DeviceBufferBase *CUDARuntime::allocateMemory(size_t bytes) {
  CUDARawDeviceBuffer raw;
  raw.allocate(bytes);
  auto wrapped = new CUDADeviceBuffer<char>(std::move(raw));
  _memory.push_back(std::unique_ptr<DeviceBufferBase>(static_cast<DeviceBufferBase*>(wrapped)));
  return wrapped;
}
}
}