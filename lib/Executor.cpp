//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/Executor.h"
#include "pacxx/detail/common/ExecutorHelper.h"
#include "pacxx/ModuleLoader.h"
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

using namespace llvm;

namespace pacxx {
namespace v2 {

std::vector<Executor> *getExecutorMemory() {
  static std::vector<Executor> *executors = new std::vector<Executor>();
  return executors; // TODO: free resources at application's exit
}

std::vector<Executor> &Executor::getExecutors() { return *getExecutorMemory(); }

Executor::Executor(std::unique_ptr<Runtime> &&rt)
    : _ctx(new LLVMContext()), _runtime(std::move(rt)) {
  core::CoreInitializer::initialize();
}

Executor::Executor(Executor &&other) {
  _ctx = std::move(other._ctx);
  _runtime = std::move(other._runtime);
  _id = other._id;
  _kernel_translation = std::move(other._kernel_translation);
}

Executor::~Executor() { __verbose("destroying executor ", _id); }

void Executor::setModule(std::unique_ptr<llvm::Module> M) {
  _runtime->link(std::move(M));

  auto &nM = _runtime->getModule();
  for (auto &F : nM.getFunctionList())
    _kernel_translation[cleanName(F.getName().str())] = F.getName().str();
}

Executor &get_executor(unsigned id) { return Executor::get(id); }

unsigned Executor::getID() { return _id; }

ExecutingDevice Executor::getExecutingDeviceType() {
  switch (_runtime->getKind()) {
#ifdef PACXX_ENABLE_CUDA
  case Runtime::RuntimeKind::RK_CUDA:
    return ExecutingDevice::GPUNvidia;
#endif
  case Runtime::RuntimeKind::RK_Native:
    return ExecutingDevice::CPU;
#ifdef PACXX_ENABLE_HIP
  case Runtime::RuntimeKind::RK_HIP:
    return ExecutingDevice::GPUAMD;
#endif
  case Runtime::RuntimeKind::RK_Remote:
    return ExecutingDevice::CPU; // FIXME
  default:
    break;
  }
  llvm_unreachable("unknown runtime kind maybe the runtime is not linked in");
}

size_t Executor::getConcurrentCores() { return _runtime->getConcurrentCores(); }

std::string Executor::cleanName(const std::string &name) {
  auto cleaned_name =
      std::regex_replace(name, std::regex("S[0-9A-Z]{0,9}_"), "");
  cleaned_name =
      std::regex_replace(cleaned_name, std::regex("5pacxx"), ""); // bad hack
  cleaned_name =
      std::regex_replace(cleaned_name, std::regex("2v2"), ""); // bad hack
  // cleaned_name = std::regex_replace(cleaned_name,
  // std::regex("S[0-9A-Z]{0,9}_"), "");
  auto It = cleaned_name.find("$_");
  if (It == std::string::npos)
    return cleaned_name;
  It += 2;
  auto value =
      std::to_string(std::strtol(&cleaned_name[It], nullptr, 10)).size();
  cleaned_name.erase(It + value);
  return cleaned_name;
}

Runtime &Executor::rt() { return *_runtime; }

void Executor::synchronize() { _runtime->synchronize(); }

std::string Executor::getFNameForLambda(std::string name) {
  std::string FName;
  const llvm::Module &M = _runtime->getModule();
  auto it = _kernel_translation.find(name);
  if (it == _kernel_translation.end()) {
    auto clean_name = cleanName(name);
    for (auto &p : _kernel_translation)
      if (p.first.find(clean_name) != std::string::npos) {
        FName = p.second;
        //_kernel_translation[name] = F.getName().str();
      }
  } else
    FName = it->second;

  auto F = M.getFunction(FName);
  if (!F) {
    throw common::generic_exception("Kernel function not found in module! " +
                                    cleanName(name));
  }
  return FName;
}

const char *__moduleStart(const char *start) {
  static const char *ptr = nullptr;
  if (!ptr)
    ptr = start;
  return ptr;
}

const char *__moduleEnd(const char *end) {
  static const char *ptr = nullptr;
  if (!ptr)
    ptr = end;
  return ptr;
}

void registerModule(const char *start, const char *end) {
  __moduleStart(start);
  __moduleEnd(end);
}
void initializeModule(Executor &exec) {
  ModuleLoader loader(exec.getLLVMContext());
	auto M =
      loader.loadInternal(__moduleStart(), __moduleEnd() - __moduleStart());
  exec.setModule(std::move(M));
}

void initializeModule(Executor &exec, const char* ptr, size_t size) {
  ModuleLoader loader(exec.getLLVMContext());
	auto M =
      loader.loadInternal(ptr, size);
  exec.setModule(std::move(M));
}

    Kernel &Executor::get_kernel_by_name(std::string name, KernelConfiguration config,
                           const void* args, size_t size, bool force_name) {
    if (!force_name)
      name = getFNameForLambda(name);

    Kernel &K = _runtime->getKernel(name);

    K.configurate(config);
    K.setLambdaPtr(args, size);

    _runtime->evaluateStagedFunctions(K);

    return K;
  }

} // namespace v2
} // namespace pacxx
