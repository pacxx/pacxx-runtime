//
// Created by mhaidl on 30/05/16.
//

#include "detail/CUDARuntime.h"
#include "detail/CUDAErrorDetection.h"
#include "detail/Log.h"
#include <cuda.h>
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

CUDARuntime::~CUDARuntime() {}

void CUDARuntime::linkMC(const std::string &MC) {
  float walltime;
  char error_log[81920];
  char info_log[81920];
  size_t logSize = 81920;

  // Setup linker options

  CUjit_option lioptions[] = {
      CU_JIT_WALL_TIME,                   // Return walltime from JIT compilation
      CU_JIT_INFO_LOG_BUFFER,             // Pass a buffer for info messages
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,  // Pass the size of the info buffer
      CU_JIT_ERROR_LOG_BUFFER,            // Pass a buffer for error message
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

  void CUDARuntime::setArguments(std::vector<char> args) {
    std::vector<void*> launch_args;
    size_t args_size = args.size();
    launch_args.push_back(CU_LAUNCH_PARAM_BUFFER_POINTER);
    launch_args.push_back(reinterpret_cast<void*>(args.data()));
    launch_args.push_back(CU_LAUNCH_PARAM_BUFFER_SIZE);
    launch_args.push_back(reinterpret_cast<void*>(&args_size));
    launch_args.push_back(CU_LAUNCH_PARAM_END);
  }
}
}