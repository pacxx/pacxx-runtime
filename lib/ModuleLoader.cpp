//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include "pacxx/ModuleLoader.h"
#include "pacxx/detail/common/Common.h"

using namespace llvm;

namespace pacxx {
namespace v2 {

std::unique_ptr<llvm::Module> ModuleLoader::loadIR(const std::string &IR) {
  SMDiagnostic Diag;
  auto mem = MemoryBuffer::getMemBuffer(IR, "llvm IR");
  return parseIR(mem->getMemBufferRef(), Diag, _ctx);
}

std::unique_ptr<llvm::Module>
ModuleLoader::loadFile(const std::string &filename) {
  SMDiagnostic Diag;
  std::string bytes = common::read_file(filename);
  auto mem = MemoryBuffer::getMemBuffer(bytes, filename);
  return parseIR(mem->getMemBufferRef(), Diag, _ctx);
}

std::unique_ptr<llvm::Module> ModuleLoader::loadInternal(const char *ptr,
                                                         size_t size) {
  SMDiagnostic Diag;
  std::string bytes(ptr, size);
  auto mem = MemoryBuffer::getMemBuffer(bytes, "internal IR");
  return parseIR(mem->getMemBufferRef(), Diag, _ctx);
}

std::unique_ptr<llvm::Module>
ModuleLoader::loadAndLink(std::unique_ptr<llvm::Module> old,
                          const std::string &filename) {

  auto loaded = loadFile(filename);

  return link(std::move(old), std::move(loaded));
}

std::unique_ptr<llvm::Module>
ModuleLoader::link(std::unique_ptr<llvm::Module> m1,
                   std::unique_ptr<llvm::Module> m2) {

  std::unique_ptr<llvm::Module> composite = std::move(m1);
  Linker linker(*composite);
  linker.linkInModule(std::move(m2));

  return std::move(composite);
}
}
}
