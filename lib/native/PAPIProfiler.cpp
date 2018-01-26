//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/native/PAPIProfiler.h"
#include "pacxx/detail/native/papipp.h"
#include "pacxx/detail/native/NativeRuntime.h"
#include "pacxx/Executor.h"
#include <fstream>
#include "pacxx/detail/common/json.hpp"

namespace pacxx {
namespace v2 {

PAPIProfiler::PAPIProfiler() {}

void PAPIProfiler::updateKernel(Kernel *kernel) {
	__verbose("PAPIProfiler updateKernel");
	_kernel = kernel;
	stats[static_cast<NativeKernel*>(_kernel)->getName()].emplace_back();
	__verbose("Current kernel run count: ", stats[static_cast<NativeKernel*>(_kernel)->getName()].size());
}

void PAPIProfiler::profile() {
	__verbose("PAPIProfiler profile");
	//for (const std::string& metricString : profilingMetrics) profileSingle(metricString);
	profileSingle("blah");
}

void PAPIProfiler::report() {
	nlohmann::json rprt(stats);
	if (OutFilePath.empty()) __message(rprt.dump(4));
	else
  {
    std::ofstream ReportFile(OutFilePath, std::ios_base::out | std::ios_base::trunc);
    if (ReportFile.is_open()) {
      ReportFile << std::setw(4) << rprt << std::endl;
      ReportFile.close();
    }
  }
}

void PAPIProfiler::profileSingle(const std::string& metricName) {
  papi::event_set<PAPI_TOT_INS> events;
  events.start_counters();
  {
		static_cast<NativeKernel*>(_kernel)->launch();
		Executor::get().restoreArgs();
	}
  events.stop_counters();
  stats[static_cast<NativeKernel*>(_kernel)->getName()].back()[metricName] = events.at<0>().counter();
}

}
}
