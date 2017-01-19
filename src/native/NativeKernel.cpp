//
// Created by lars on 07/10/16.
//

#include "detail/native/NativeKernel.h"
#include <detail/native/NativeRuntime.h>
#include <detail/common/Log.h>
#include <detail/common/Exceptions.h>
#include <detail/common/Timing.h>
#include <tbb/tbb.h>

namespace pacxx {
  namespace v2 {

    NativeKernel::NativeKernel(NativeRuntime &runtime, void *fptr) :
        _runtime(runtime),
        _fptr(fptr),
        _staged_values_changed(false),
        _disable_staging(false),
        _hostArgBufferSize(0) {}

    NativeKernel::~NativeKernel() {}

    void NativeKernel::configurate(KernelConfiguration config) {
        if (_config != config) {
            _config = config;
        }
    }

    KernelConfiguration NativeKernel::getConfiguration() const { return _config; }

    void NativeKernel::setArguments(const std::vector<char> &arg_buffer) {
        __verbose("Set kernel args");
        _args = arg_buffer;
    }

    const std::vector<char>& NativeKernel::getArguments() const { return _args; }

    void NativeKernel::setHostArguments(const std::vector<char> &arg_buffer) {
        _host_args = arg_buffer;
    }

    const std::vector<char>& NativeKernel::getHostArguments() const { return _host_args; }

      void NativeKernel::launch() {

        if (!_fptr || _staged_values_changed) { // kernel has no function ptr yet. request kernel transformation and recompilation if necessary
            _runtime.requestIRTransformation(*this);
            _staged_values_changed = false;
        }

          __verbose("Launching kernel: \nblocks(", _config.blocks.x, ",",
                _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
                _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,
                ")\nshared_mem=", _config.sm_size);



          // The kernel wrapper always has this function signature.
          // The kernel args are constructed from the char buffer
          auto functor = reinterpret_cast<void (*) (int, int, int,
                                                int, int, int, int, char*)>(_fptr);

          std::chrono::high_resolution_clock::time_point start, end;
          unsigned runs = 1;

          start = std::chrono::high_resolution_clock::now();

          for(unsigned i = 0; i < runs; ++i) {
              tbb::parallel_for(size_t(0), _config.blocks.z, [&](size_t bidz) {
                  tbb::parallel_for(size_t(0), _config.blocks.y, [&](size_t bidy) {
                      tbb::parallel_for(size_t(0), _config.blocks.x, [&](size_t bidx) {
                          functor(bidx, bidy, bidz, _config.threads.x, _config.threads.y,
                                  _config.threads.z, _config.sm_size,  _args.data());
                      });
                  });
              });
          }

          end = std::chrono::high_resolution_clock::now();

          auto  time_tbb = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

          __verbose("Time measured in runtime (TBB) : ", time_tbb / runs, " us (", runs, " iterations)");


          /*
          start = std::chrono::high_resolution_clock::now();

          for(unsigned i = 0; i < runs; ++i) {
#pragma omp parallel for collapse(3) schedule(dynamic)
              for(unsigned bidz = 0; bidz < _config.blocks.z; ++bidz)
                  for(unsigned bidy = 0; bidy < _config.blocks.y; ++bidy)
                      for(unsigned bidx = 0; bidx < _config.blocks.x; ++bidx)
                          functor(bidx, bidy, bidz, _config.threads.x, _config.threads.y,
                                  _config.threads.z, _config.sm_size, _args.data());
          }

          end = std::chrono::high_resolution_clock::now();

          auto  time_omp = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

          __verbose("Time measured in runtime (OMP) : ", time_omp / runs, " us (", runs, " iterations)");
          */

          if(_callback)
            _callback();
      }

      void NativeKernel::setStagedValue(int ref, long long value, bool inScope) {
        auto old = _staged_values[ref];
        if (old != value) {
            _staged_values[ref] = value;
            if (inScope)
                _staged_values_changed = true;
        }
      }

      const std::map<int, long long>& NativeKernel::getStagedValues() const {
          return _staged_values;
      }

      void NativeKernel::setName(std::string name) {
          _name = name;
      }

      const std::string& NativeKernel::getName() const { return _name; }

      void NativeKernel::disableStaging() {
          _disable_staging = true;
      }

      bool NativeKernel::requireStaging() { return !_disable_staging; }

      size_t NativeKernel::getHostArgumentsSize() const {
        return _hostArgBufferSize;
      }

      void NativeKernel::setHostArgumentsSize(size_t size) {
        _hostArgBufferSize = size;
      }
  }
}
