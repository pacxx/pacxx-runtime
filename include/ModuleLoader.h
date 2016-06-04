//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_MODULELOADER_H
#define PACXX_V2_MODULELOADER_H

#include <memory>

namespace llvm {
  class Module;
}

namespace pacxx {
  namespace v2 {

    class ModuleLoader
    {
    public:
      std::unique_ptr<llvm::Module> loadFile(const std::string& filename);
      std::unique_ptr<llvm::Module> loadInternal(const char* ptr, size_t size);
    };

  }
}
#endif //PACXX_V2_MODULELOADER_H
