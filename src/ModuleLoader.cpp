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

    std::unique_ptr<llvm::Module> ModuleLoader::loadIR(const std::string& IR) {
      SMDiagnostic Diag;
      auto mem = MemoryBuffer::getMemBuffer(IR, "llvm IR");
      return parseIR(mem->getMemBufferRef(), Diag, getGlobalContext());
    }

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

      auto loaded = loadFile(filename);

      return link(std::move(old), std::move(loaded));
    }

    std::unique_ptr<llvm::Module> ModuleLoader::link(std::unique_ptr<llvm::Module> m1,
                                                     std::unique_ptr<llvm::Module> m2) {

      std::unique_ptr<llvm::Module> composite = std::move(m1);
      Linker linker(composite.get());
      linker.linkInModule(m2.get());

      return std::move(composite);
    }

  }
}