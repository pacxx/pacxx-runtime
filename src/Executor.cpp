#include "pacxx/Executor.h"

namespace pacxx {
namespace v2 {

Executor::Executor(std::unique_ptr<IRRuntime> &&rt) :
    _ctx(new LLVMContext()), _runtime(std::move(rt)),
    _mem_manager(*_runtime) {
  core::CoreInitializer::initialize();
}

Executor::Executor(Executor &&other) : _mem_manager(std::move(other._mem_manager)) {
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

unsigned Executor::getID() { return _id; };

void Executor::setMSPModule(std::unique_ptr<llvm::Module> M) {
  _runtime->initializeMSP(std::move(M));
}

ExecutingDevice Executor::getExecutingDeviceType() {
  switch (_runtime->getRuntimeType()) {
#ifdef PACXX_ENABLE_CUDA
  case RuntimeType::CUDARuntimeTy:return ExecutingDevice::GPUNvidia;
#endif
  case RuntimeType::NativeRuntimeTy:return ExecutingDevice::CPU;
  }
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

RawDeviceBuffer &Executor::allocateRaw(size_t bytes) {
  __verbose("allocating raw memory: ", bytes);
  return *_runtime->allocateRawMemory(bytes);
}

void Executor::freeRaw(RawDeviceBuffer &buffer) { _runtime->deleteRawMemory(&buffer); }

MemoryManager &Executor::mm() { return _mem_manager; }

IRRuntime &Executor::rt() { return *_runtime; }

void Executor::synchronize() { _runtime->synchronize(); }

llvm::legacy::PassManager &Executor::getPassManager() { return _runtime->getPassManager(); }

}
}

