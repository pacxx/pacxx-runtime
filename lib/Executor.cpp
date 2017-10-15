#include "pacxx/Executor.h"

namespace pacxx {
namespace v2 {

Executor::Executor(std::unique_ptr<IRRuntime> &&rt) :
    _ctx(new LLVMContext()), _runtime(std::move(rt)){
  core::CoreInitializer::initialize();
}

Executor::Executor(Executor &&other)  {
  _ctx = std::move(other._ctx);
  _runtime = std::move(other._runtime);
  _id = other._id;
  _kernel_translation = std::move(other._kernel_translation);
}

void Executor::setModule(std::unique_ptr <llvm::Module> M) {

  _runtime->link(std::move(M));

  auto &nM = _runtime->getModule();
  for (auto &F : nM.getFunctionList())
    _kernel_translation[cleanName(F.getName().str())] = F.getName().str();
}

Executor &get_executor(unsigned id) {
  return Executor::get(id);
}

unsigned Executor::getID() { return _id; }

ExecutingDevice Executor::getExecutingDeviceType() {
  switch (_runtime->getKind()) {
#ifdef PACXX_ENABLE_CUDA
  case IRRuntime::RuntimeKind::RK_CUDA: return ExecutingDevice::GPUNvidia;
#endif
  case IRRuntime::RuntimeKind::RK_Native: return ExecutingDevice::CPU;
  }
  llvm_unreachable("unknown runtime kind");
}

size_t Executor::getConcurrentCores() {
  return _runtime->getConcurrentCores();
}

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

RawDeviceBuffer &Executor::allocateRaw(size_t bytes, MemAllocMode mode) {
  __verbose("allocating raw memory: ", bytes);
  return *_runtime->allocateRawMemory(bytes, mode);
}

void Executor::freeRaw(RawDeviceBuffer &buffer) { _runtime->deleteRawMemory(&buffer); }

IRRuntime &Executor::rt() { return *_runtime; }

void Executor::synchronize() { _runtime->synchronize(); }

#define __PACXX_RUNTIME_LINKING
#ifdef __PACXX_RUNTIME_LINKING
const char *llvm_start = nullptr;
const char *llvm_end = nullptr;
#endif

Executor &Executor::Create(std::unique_ptr<IRRuntime> rt, std::string module_bytes) {
  auto &executors = getExecutors();

  executors.emplace_back(std::move(rt));
  auto &instance = executors.back();

  instance._id = executors.size() - 1;
  __verbose("Created new Executor with id: ", instance.getID());
  ModuleLoader loader(instance.getLLVMContext());
  if (module_bytes == "") {
    auto M = loader.loadInternal(llvm_start, llvm_end - llvm_start);
    instance.setModule(std::move(M));
  } else {
    ModuleLoader loader(instance.getLLVMContext());
    auto M = loader.loadInternal(module_bytes.data(), module_bytes.size());
    instance.setModule(std::move(M));
  }

  return instance;
}


}
}

