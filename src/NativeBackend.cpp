//
// Created by mhaidl on 14/06/16.
//

#include "detail/native/NativeBackend.h"


namespace pacxx
{
  namespace v2
  {
    NativeBackend::NativeBackend() { }

    NativeBackend::~NativeBackend() { }

    std::string NativeBackend::compile(llvm::Module &M) {
      return std::string();
    }
  }
}
