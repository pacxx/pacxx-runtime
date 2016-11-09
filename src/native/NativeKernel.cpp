//
// Created by lars on 07/10/16.
//

#include "detail/native/NativeKernel.h"
#include <detail/native/NativeRuntime.h>
#include <detail/common/Log.h>
#include <detail/common/Exceptions.h>

namespace pacxx {
  namespace v2 {

    NativeKernel::NativeKernel(NativeRuntime &runtime, void *fptr) :
        _runtime(runtime),
        _fptr(fptr),
        _staged_values_changed(false),
        _disable_staging(false) {}

    NativeKernel::~NativeKernel() {}

    void NativeKernel::configurate(KernelConfiguration config) {
      if (_config != config) {
        _config = config;
      }
    }

    KernelConfiguration NativeKernel::getConfiguration() const { return _config; }

      void NativeKernel::setArguments(const std::vector<char> &arg_buffer) {
          _args = arg_buffer;
      }

      const std::vector<char>& NativeKernel::getArguments() const { return _args; }

      void NativeKernel::setHostArguments(const std::vector<char> &arg_buffer) {
          _host_args = arg_buffer;
      }

      const std::vector<char>& NativeKernel::getHostArguments() const { return _host_args; }

      void NativeKernel::launch() {
          __verbose("Launching kernel: \nblocks(", _config.blocks.x, ",",
                _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
                _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,")");

          std::chrono::high_resolution_clock::time_point start, end;

          start = std::chrono::high_resolution_clock::now();

          for(size_t bidz = 0; bidz < _config.blocks.z; ++bidz)
            for(size_t bidy = 0; bidy < _config.blocks.y; ++bidy)
                for(size_t bidx = 0; bidx < _config.blocks.x; ++bidx)
                      _runtime.runOnThread(_fptr, bidx, bidy, bidz, _config.threads.x, _config.threads.y,
                                           _config.threads.z, _args.data());
          _runtime.synchronize();

          end = std::chrono::high_resolution_clock::now();

          auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

          __verbose("Time measured in runtime :", time, " us");
      }

      void NativeKernel::setStagedValue(int ref, long long value, bool inScope) { throw new common::generic_exception("not supported"); }

      const std::map<int, long long>& NativeKernel::getStagedValues() const { throw new common::generic_exception("not supported"); }

      void NativeKernel::setName(std::string name) {
          _name = name;
      }

      const std::string& NativeKernel::getName() const { return _name; }

      void NativeKernel::disableStaging() {
          _disable_staging = true;
      }

      bool NativeKernel::requireStaging() { return !_disable_staging; }
  }
}
