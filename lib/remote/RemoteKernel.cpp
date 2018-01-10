//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/remote/RemoteKernel.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/remote/RemoteRuntime.h"
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

namespace pacxx {
namespace v2 {

RemoteKernel::RemoteKernel(RemoteRuntime &runtime, std::string name)
    : Kernel(runtime, name), _runtime(runtime) {}

RemoteKernel::~RemoteKernel() {}

void RemoteKernel::configurate(KernelConfiguration config) {
  if (_config != config) {
    _config = config;
}
}

void RemoteKernel::launch() {
  __debug("Launching kernel: ", _name);
  __verbose("Kernel configuration: \nblocks(", _config.blocks.x, ",",
            _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
            _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,
            ")\nshared_mem=", _config.sm_size);

  _runtime.launchRemoteKernel(_name, _lambdaPtr, _lambdaSize, _config);
}


}
}
