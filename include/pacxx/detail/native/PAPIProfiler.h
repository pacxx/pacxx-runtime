//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_PAPIPROFILER_H
#define PACXX_V2_PAPIPROFILER_H

#include "../Profiler.h"
#include "NativeKernel.h"
#include <list>

namespace pacxx {
namespace v2 {

class PAPIProfiler : public Profiler {
public:
  PAPIProfiler();

  virtual ~PAPIProfiler() {};

  virtual void updateKernel(Kernel *kernel) override;

  virtual void profile() override;

  virtual void report() override;

private:
  void profileSingle(const std::string& metricName);

  std::list<std::string> profilingMetrics;

  ///      Kernel name, run count,           metric name, value
  std::map<std::string, std::vector<std::map<std::string, long long>>> stats;
};
}
}

#endif // PACXX_V2_PAPIPROFILER_H

