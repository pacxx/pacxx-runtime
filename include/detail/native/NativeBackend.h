//
// Created by mhaidl on 14/06/16.
//

#ifndef PACXX_V2_NATIVEBACKEND_H
#define PACXX_V2_NATIVEBACKEND_H

#include <detail/IRCompiler.h>

namespace pacxx
{
  namespace v2
  {
    class NativeBackend : public IRCompiler {
    public:
      NativeBackend();
      virtual ~NativeBackend();

      virtual std::string compile(llvm::Module &M) override;
    };
  }
}
#endif //PACXX_V2_NATIVEBACKEND_H
