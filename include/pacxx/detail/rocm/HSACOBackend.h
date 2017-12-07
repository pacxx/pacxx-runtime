//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_HSACOBACKEND_H
#define PACXX_V2_HSACOBACKEND_H

namespace llvm {
class Module;
class Target;
class TargetOptions;
class TargetMachine;
} // namespace llvm

namespace pacxx {
namespace v2 {
class HSACOBackend {
public:
  HSACOBackend();

  ~HSACOBackend() {}

  void initialize(unsigned gfx);

  std::unique_ptr<llvm::Module> prepareModule(llvm::Module &M);

  std::string compile(llvm::Module &M);

private:
  const llvm::Target *_target;
  std::unique_ptr<llvm::TargetMachine> _machine;
  std::string _cpu, _features;
  unsigned _gcnArch;
};
} // namespace v2
} // namespace pacxx

#endif // PACXX_V2_HSACOBACKEND_H
