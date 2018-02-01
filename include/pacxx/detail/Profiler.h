//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_PROFILER_H
#define PACXX_V2_PROFILER_H

#include <memory>
#include <string>
#include "Kernel.h"

namespace pacxx {
namespace v2 {

class Profiler {
public:
  Profiler();

  virtual ~Profiler() {};

  virtual bool enabled();

  virtual bool preinit(void* settings) { return true; };

  virtual bool postinit(void* settings) { return true; };

  virtual void updateKernel(Kernel *kernel);

  virtual void dryrun() {};

  virtual void profile() {};

  virtual void report() {};

protected:
  static Kernel* _kernel;
  std::string InFilePath, OutFilePath;
  bool _enabled;
};

}
}

#endif // PACXX_V2_PROFILER_H
