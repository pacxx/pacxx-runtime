//
// Created by lars on 07/10/16.
//

#include "detail/native/NativeKernel.h"
#include <detail/native/NativeRuntime.h>
#include <detail/common/Log.h>
#include <detail/common/Exceptions.h>

namespace pacxx {
  namespace v2 {

    NativeKernel::NativeKernel(NativeRuntime &runtime, llvm::Function *function) :
        _runtime(runtime),
        _function(function),
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
      }

      const std::vector<char>& NativeKernel::getArguments() const { return _args; }

      void NativeKernel::setHostArguments(const std::vector<char> &arg_buffer) {
          _host_args = arg_buffer;
      }

      const std::vector<char>& NativeKernel::getHostArguments() const { return _host_args; }

      //TODO launch multiple threads
      void NativeKernel::launch() {
          __verbose("Launching kernel: \nblocks(", _config.blocks.x, ",",
                _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
                _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,")");
          //TODO create kernel arguments vector
          for(size_t bidx = 0; bidx < _config.blocks.x; ++bidx)
              for(size_t bidy = 0; bidy < _config.blocks.y; ++bidy)
                  for(size_t bidz = 0; bidz < _config.blocks.z; ++bidz)
                      _runtime.runOnThread(_function, prepareFunctionArgs(bidx, bidy, bidz));
      }

      std::vector<llvm::GenericValue> NativeKernel::prepareFunctionArgs(const size_t bidx, size_t bidy, size_t bidz) {
          llvm::FunctionType *type = _function->getFunctionType();
          size_t numArgs = type->getNumParams();
          __verbose("function has ", numArgs, " arguments");
          type->dump();
          //TODO refactor if working
          std::vector<GenericValue> args(type->getNumParams());
          args[0].IntVal = bidx;
          args[1].IntVal = bidy;
          args[2].IntVal = bidz;
          args[3].IntVal = _config.threads.x;
          args[4].IntVal = _config.threads.y;
          args[5].IntVal = _config.threads.z;
          args[6].PointerVal = _args.data();
          __verbose(numArgs);
          __verbose(args.size());
          if(args.size() != numArgs)
              throw new common::generic_exception("failed to create function arguments");
          return args;
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
