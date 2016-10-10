//
// Created by lars on 07/10/16.
//

#include "detail/native/NativeKernel.h"
#include <detail/native/NativeRuntime.h>
#include <detail/common/Log.h>
#include <detail/common/Exceptions.h>

namespace pacxx {
  namespace v2 {

    NativeKernel::NativeKernel(NativeRuntime &runtime, llvm::FunctionType* type, void *fptr) :
        _runtime(runtime),
        _type(type),
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
          _args_size = _args.size();
          _launch_args.push_back(reinterpret_cast<int *>(_args.data()));
      }

      const std::vector<char>& NativeKernel::getArguments() const { return _args; }

      void NativeKernel::setHostArguments(const std::vector<char> &arg_buffer) {
          _host_args = arg_buffer;
      }

      const std::vector<char>& NativeKernel::getHostArguments() const { return _host_args; }

      //TODO reinterpret_cast to match kernel launch args
      //TODO launch multiple threads
      void NativeKernel::launch() {
          __verbose(_args_size);
          if(!_fptr)
              throw new common::generic_exception("kernel has no function ptr");
          __verbose("Launching kernel: \nblocks(", _config.blocks.x, ",",
                _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
                _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,")");
          __verbose(_type->getNumParams());
          _type->dump();
          auto functor = reinterpret_cast<void *(*)(size_t, size_t, size_t, size_t, size_t, size_t, int**, int** , int** , int**)> (_fptr);

          functor(0, 0, 0, _config.threads.x, _config.threads.y, _config.threads.z, &_launch_args[0],
                  &_launch_args[1]+sizeof(int*), &_launch_args[2], &_launch_args[3]);
          //TODO run on multiple threads
          /*_runtime.runFunctionOnThread<void *(*)(size_t, size_t, size_t, size_t, size_t, size_t, void** , void** , void**)>
                  (functor, 0, 0, 0, _config.threads.x, _config.threads.y, _config.threads.z,
                                       &_args[0], &_args[1], &_args[2]);
         */
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
