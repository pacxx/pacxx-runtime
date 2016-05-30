//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_CUDARUNTIME_H
#define PACXX_V2_CUDARUNTIME_H
#include <string>
#include <map>
#include "IRRuntime.h"

// forward declarations of cuda driver structs
struct CUctx_st;
typedef struct CUctx_st* CUcontext;
struct CUmod_st;
typedef struct CUmod_st* CUmodule;

namespace pacxx
{
  namespace v2
  {
    class CUDARuntime : public IRRuntime
    {
    public:
      CUDARuntime(unsigned dev_id);
      virtual ~CUDARuntime();

      virtual void linkMC(const std::string& MC) override;
      virtual void setArguments(std::vector<char> args) override;

    private:
      CUcontext _context;
      CUmodule _mod;
    };
  }
}

#endif //PACXX_V2_CUDARUNTIME_H
