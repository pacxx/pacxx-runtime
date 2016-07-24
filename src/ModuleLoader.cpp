//
// Created by mhaidl on 29/05/16.
//

#include <memory>

#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>

#include "ModuleLoader.h"
#include "detail/common/Common.h"

using namespace llvm;

namespace pacxx
{
  namespace v2
  {
    std::unique_ptr<llvm::Module> ModuleLoader::loadFile(const std::string& filename)
    {
      SMDiagnostic Diag;
      std::string bytes = common::read_file(filename);
      auto mem = MemoryBuffer::getMemBuffer(bytes, filename);
      return parseIR(mem->getMemBufferRef(), Diag, getGlobalContext());
    }

    std::unique_ptr<llvm::Module> ModuleLoader::loadInternal(const char* ptr, size_t size)
    {
      SMDiagnostic Diag;
      std::string bytes(ptr, size);
      auto mem = MemoryBuffer::getMemBuffer(bytes, "internal IR");
      return parseIR(mem->getMemBufferRef(), Diag, getGlobalContext());
    }

    std::unique_ptr<llvm::Module> ModuleLoader::loadAndLink(std::unique_ptr<llvm::Module> old,
                                                            const std::string& filename) {

      std::unique_ptr<llvm::Module> composite = std::move(old);
      Linker linker(composite.get());

      auto loaded = loadFile(filename);
      linker.linkInModule(loaded.get());

      return std::move(composite);
    }

  }
}