//
// Created by mhaidl on 31/05/16.
//
#include "pacxx/detail/cuda/CUDAKernel.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/cuda/CUDAErrorDetection.h"
#include "pacxx/detail/cuda/CUDARuntime.h"
#include <cuda.h>

namespace pacxx {
namespace v2 {

CUDAKernel::CUDAKernel(CUDARuntime &runtime, CUfunction fptr)
    : _runtime(runtime), _fptr(fptr), _staged_values_changed(false),
      _disable_staging(false), _hostArgBufferSize(0) {}

CUDAKernel::~CUDAKernel() {}

void CUDAKernel::configurate(KernelConfiguration config) {
  if (_config != config) {
    _config = config;
    std::vector<size_t> a(6);
    a[0] = config.threads.x;
    a[1] = config.threads.y;
    a[2] = config.threads.z;
    a[3] = config.blocks.x;
    a[4] = config.blocks.y;
    a[5] = config.blocks.z;

    for (size_t i = 0; i < a.size(); ++i) {
      // setStagedValue((i * -1) - 1, a[i], true);
    }
  }
}

KernelConfiguration CUDAKernel::getConfiguration() const { return _config; }

void CUDAKernel::setArguments(const std::vector<char> &arg_buffer) {
  _args = arg_buffer;
  _args_size = _args.size();
  _launch_args.clear(); // remove old args first
  _launch_args.push_back(CU_LAUNCH_PARAM_BUFFER_POINTER);
  _launch_args.push_back(reinterpret_cast<void *>(_args.data()));
  _launch_args.push_back(CU_LAUNCH_PARAM_BUFFER_SIZE);
  _launch_args.push_back(reinterpret_cast<void *>(&_args_size));
  _launch_args.push_back(CU_LAUNCH_PARAM_END);
}

const std::vector<char> &CUDAKernel::getArguments() const { return _args; }

void CUDAKernel::setHostArguments(const std::vector<char> &arg_buffer) {
  _host_args = arg_buffer;
}

const std::vector<char> &CUDAKernel::getHostArguments() const {
  return _host_args;
}

void CUDAKernel::launch() {
  if (!_fptr || _staged_values_changed) { // kernel has no function ptr yet.
                                          // request kernel transformation and
                                          // recompilation if necessary
    _runtime.requestIRTransformation(*this);
    _staged_values_changed = false;
  }
  __verbose("Launching kernel: \nblocks(", _config.blocks.x, ",",
            _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
            _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,
            ")\nshared_mem=", _config.sm_size);

  SEC_CUDA_CALL(cuLaunchKernel(
      _fptr, _config.blocks.x, _config.blocks.y, _config.blocks.z,
      _config.threads.x, _config.threads.y, _config.threads.z, _config.sm_size,
      NULL, nullptr, &_launch_args[0]));
  if (_callback)
    SEC_CUDA_CALL(cudaStreamAddCallback(nullptr, CUDARuntime::fireCallback,
                                        &_callback, 0));
}

void CUDAKernel::setStagedValue(int ref, long long value, bool inScope) {
  auto old = _staged_values[ref];
  if (old != value) {
    _staged_values[ref] = value;
    if (inScope)
      _staged_values_changed = true;
  }
}

const std::map<int, long long> &CUDAKernel::getStagedValues() const {
  return _staged_values;
}

void CUDAKernel::setName(std::string name) { _name = name; }

const std::string &CUDAKernel::getName() const { return _name; }

void CUDAKernel::disableStaging() { _disable_staging = true; }

bool CUDAKernel::requireStaging() { return !_disable_staging; }

size_t CUDAKernel::getHostArgumentsSize() const { return _hostArgBufferSize; }

void CUDAKernel::setHostArgumentsSize(size_t size) {
  _hostArgBufferSize = size;
}
}
}