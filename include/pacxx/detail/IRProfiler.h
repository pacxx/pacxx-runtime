//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_IRPROFILER_H
#define PACXX_V2_IRPROFILER_H

#include <memory>
#include "Kernel.h"

namespace pacxx {
namespace v2 {

class IRProfiler {
public:
  IRProfiler();

  virtual ~IRProfiler() {};

  virtual bool enabled();

  virtual bool preinit(void* settings) = 0;

  virtual void postinit(void* settings) = 0;

  virtual void updateKernel(Kernel *kernel);

  virtual void dryrun() = 0;

  virtual void profile() = 0;

  virtual void report() = 0;

protected:
  static Kernel* _kernel;
  std::string InFilePath, OutFilePath;
  bool _enabled;
};

}
}

#endif // PACXX_V2_IRPROFILER_H
