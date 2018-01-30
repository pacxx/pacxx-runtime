//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

extern "C"
{
  #include <papi.h>
}

#include "pacxx/detail/native/PAPIProfiler.h"
#include "pacxx/detail/native/NativeRuntime.h"
#include "pacxx/Executor.h"
#include <fstream>
#include "pacxx/detail/common/json.hpp"

#define event_code int
#define papi_counter long long

namespace pacxx {
namespace v2 {
namespace papi {
  std::string get_event_code_name(event_code code)
  {
      std::array<char, PAPI_MAX_STR_LEN> event_name;
      ::PAPI_event_code_to_name(code, event_name.data());

      return event_name.data();
  }

  event_code get_event_code(std::string name)
  {
      event_code event;
      std::vector<char> rawname(name.c_str(), name.c_str() + name.size() + 1);
      ::PAPI_event_name_to_code(rawname.data(), &event);

      return event;
  }

  bool query_named_event(std::string name)
  {
      int ret;
      std::vector<char> rawname(name.c_str(), name.c_str() + name.size() + 1);
      __verbose("Checking ", rawname.data());
      if ((ret = ::PAPI_query_named_event(rawname.data())) != PAPI_OK)
          return false;
      else return true;
  }

  class event
  {
    public:
      event(event_code Event, papi_counter counter = {}) : _Event(Event), _counter(counter) {}

      papi_counter counter() const { return _counter; }

      event_code code() { return _Event; }
      std::string name() const { return get_event_code_name(_Event); }

    private:
      const event_code _Event;
      papi_counter _counter;
  };

  class event_set
  {
    public:
      event_set(std::vector<event_code> event_codes) : s_events(event_codes), size(event_codes.size())
      {
        _counters.resize(size);
      }
      event_set(event_code event_codes)
      {
        s_events.push_back(event_codes);
        size = 1;
        _counters.resize(size);
      }

      void start_counters()
      {
        int ret;

        if ((ret = ::PAPI_start_counters(s_events.data(), size)) != PAPI_OK)
            throw common::generic_exception(common::to_string("PAPI_start_counters failed with error: ", PAPI_strerror(ret)));

      }

      void reset_counters()
      {
        int ret;

        if ((ret = ::PAPI_read_counters(_counters.data(), size)) != PAPI_OK)
            throw common::generic_exception(common::to_string("PAPI_read_counters failed with error: ", PAPI_strerror(ret)));
      }

      void stop_counters()
      {
        int ret;

        if ((ret = ::PAPI_stop_counters(_counters.data(), size)) != PAPI_OK)
            throw common::generic_exception(common::to_string("PAPI_stop_counters failed with error: ", PAPI_strerror(ret)));
      }

      event at(std::size_t _EventIndex) const
      {
        event_code code = s_events[_EventIndex];
        return event(code, _counters[_EventIndex]);
      }

      event get(event_code _EventCode) const
      {
        auto eventIndex = std::find(s_events.begin(), s_events.end(), _EventCode);
        if (eventIndex == s_events.end())
          throw common::generic_exception(
            common::to_string("PAPI (profiler api) error: EventCode ", _EventCode, " not present in this event_set"));
        return at(std::distance(s_events.begin(), eventIndex));
      }

    private:
      std::vector<event_code> s_events;
      std::vector<papi_counter> _counters;
      std::size_t size;
  };
}

PAPIProfiler::PAPIProfiler() {}

bool PAPIProfiler::preinit(void* settings) {
  int ret;
  ret = ::PAPI_library_init(PAPI_VER_CURRENT);
  if (ret != PAPI_VER_CURRENT) {
    __fatal("PAPI library init error!");
    return false;
  }
  else return true;
}

bool PAPIProfiler::postinit(void* settings) {
  std::ifstream profiles(InFilePath);
	if (profiles.is_open()) {
		for (std::string metricString; std::getline(profiles, metricString); ) profilingMetrics.push_back(metricString);
		profiles.close();
	}
	return true;
}

void PAPIProfiler::updateKernel(Kernel *kernel) {
	__verbose("PAPIProfiler updateKernel");
	_kernel = kernel;
	stats[static_cast<NativeKernel*>(_kernel)->getName()].emplace_back();
	__verbose("Current kernel run count: ", stats[static_cast<NativeKernel*>(_kernel)->getName()].size());
}

void PAPIProfiler::dryrun() {
	__verbose("PAPIProfiler dryrun");
	std::chrono::high_resolution_clock::time_point start, end;
  start = std::chrono::high_resolution_clock::now();
  static_cast<NativeKernel*>(_kernel)->launch();
  end = std::chrono::high_resolution_clock::now();
	Executor::get().restoreArgs();
	{
	  std::stringstream foo;
	  foo << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	  foo << "us";
	  stats[static_cast<NativeKernel*>(_kernel)->getName()].back()["kernelDuration"] = foo.str();
	}
	__debug("Kernel run time: ", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(), "us");
}

void PAPIProfiler::profile() {
	__verbose("PAPIProfiler profile");
	for (const std::string& metricString : profilingMetrics) profileSingle(metricString);
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
  if (papi::query_named_event(metricName))
  {
    papi::event_set events(papi::get_event_code(metricName));
    events.start_counters();
    static_cast<NativeKernel*>(_kernel)->launch();
    events.stop_counters();
    stats[static_cast<NativeKernel*>(_kernel)->getName()].back()[events.at(0).name()] = std::to_string(events.at(0).counter());
    Executor::get().restoreArgs();
  }
}

}
}
