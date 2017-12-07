//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_PTXBACKEND_H
#define PACXX_V2_PTXBACKEND_H

namespace llvm {
	class Module;
	class Target;
	class TargetOptions;
	class TargetMachine;
} // namespace llvm

namespace pacxx {
namespace v2 {
class PTXBackend {
public:
  PTXBackend();

  ~PTXBackend();

  void initialize(unsigned CC);

  std::unique_ptr<llvm::Module> prepareModule(llvm::Module &M);

  std::string compile(llvm::Module &M);

private:
  const llvm::Target *_target;
  std::unique_ptr<llvm::TargetMachine> _machine;
  std::string _cpu, _features;
};
}
}

#endif // PACXX_V2_PTXBACKEND_H
