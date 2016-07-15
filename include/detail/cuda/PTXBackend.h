//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_PTXBACKEND_H
#define PACXX_V2_PTXBACKEND_H

#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
namespace llvm {
class Module;
class Target;
}

namespace pacxx {
namespace v2 {
  class PTXBackend {
public:
  PTXBackend();

    ~PTXBackend() { }

    void initialize();

    std::string compile(llvm::Module& M);

private:
  const llvm::Target *_target;
  llvm::TargetOptions _options;
  std::unique_ptr<llvm::TargetMachine> _machine;
  std::string _cpu, _features;
};
}
}

#endif // PACXX_V2_PTXBACKEND_H
