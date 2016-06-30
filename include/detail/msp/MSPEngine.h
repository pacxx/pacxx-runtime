//
// Created by mhaidl on 30/06/16.
//

#ifndef PACXX_V2_MSPENGINE_H
#define PACXX_V2_MSPENGINE_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <set>
#include <detail/Kernel.h>
#include <detail/common/Meta.h>

#include "../KernelArgument.h"

namespace pacxx
{
namespace v2
{
  class MSPEngine
  {
  public:

    using stub_ptr_t = void (*)(void *, void *);
    using i64_stub_ptr_t = size_t (*)(size_t);


    void initialize(std::unique_ptr<llvm::Module> M);
    void evaluate(const llvm::Function& KF, Kernel& kernel);
    void transformModule(llvm::Module& M, Kernel& K);

  private:
    llvm::ExecutionEngine* _engine;
    std::set<std::pair<llvm::Function*, int>> _stubs;

    std::map<unsigned, stub_ptr_t> FStubs;
    std::map<unsigned, i64_stub_ptr_t> i64FStubs;
  };
}

  namespace meta
  {
    struct msp_memory_translation
    {
      template <typename T, std::enable_if_t<is_vector<T>::value>* = nullptr>
      auto operator() (const T& data)
      {
        return reinterpret_cast<const char*>(&data) + 32; // TODO: Make 32 dynamic
      }

      template <typename T, std::enable_if_t<!(is_vector<T>::value || is_devbuffer<T>::value)>* = nullptr>
      auto operator() (const T& data)
      {
        return data;
      }
    };
  }

}
#endif //PACXX_V2_MSPENGINE_H
