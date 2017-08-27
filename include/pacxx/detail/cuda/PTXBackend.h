//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_PTXBACKEND_H
#define PACXX_V2_PTXBACKEND_H

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>
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

  ~PTXBackend() {}

  void initialize(unsigned CC);

  std::unique_ptr<llvm::Module> prepareModule(llvm::Module &M);

  llvm::legacy::PassManager &getPassManager();

  std::string compile(llvm::Module &M);

private:
  const llvm::Target *_target;
  llvm::TargetOptions _options;
  std::unique_ptr<llvm::TargetMachine> _machine;
  std::string _cpu, _features;
  llvm::legacy::PassManager _PM;
  llvm::SmallString<128> _ptxString;
  bool _pmInitialized;
};
}
}

#endif // PACXX_V2_PTXBACKEND_H
