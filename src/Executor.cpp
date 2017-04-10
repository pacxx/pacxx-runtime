#include "pacxx/Executor.h"

namespace pacxx {
namespace v2 {

void Executor::setModule(std::unique_ptr <llvm::Module> M) {

  _runtime->link(std::move(M));

  auto &nM = _runtime->getModule();
  for (auto &F : nM.getFunctionList())
    _kernel_translation[cleanName(F.getName().str())] = F.getName().str();
}

Executor &get_executor(unsigned id) {
  return Executor::get(id);
}
}
}

