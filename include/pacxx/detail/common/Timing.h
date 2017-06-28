//
// Created by mhaidl on 14/06/16.
//

#ifndef PACXX_V2_TIMING_H
#define PACXX_V2_TIMING_H
#include "Common.h"
#include "Log.h"
#include <chrono>
#include <type_traits>
#include <utility>

#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1, s2)

#ifdef __COUNTER__
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __COUNTER__)
#else
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)
#endif

#define SCOPED_TIMING                                                          \
  auto ANONYMOUS_VARIABLE(TIMING) =                                            \
      pacxx::common::ScopedTimingProxy(__FILE__, __LINE__, __func__) + [&]()

namespace pacxx {
namespace common {
template <typename FunctionType> struct ScopedTiming {
  std::chrono::high_resolution_clock::time_point start, end;
  FunctionType function;
  std::string message;

  ScopedTiming(FunctionType &&f, const std::string &message)
      : function(f), message(message) {
    start = std::chrono::high_resolution_clock::now();
    function();
    end = std::chrono::high_resolution_clock::now();
  }

  ~ScopedTiming() {
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
#ifndef __PACXX_TIMING_STANDALONE__
    __verbose(message, " timed: ", time, "us");
#else
    std::cout << common::to_string(message, " timed: ", time, "us")
              << std::endl;
#endif
  }
};

struct ScopedTimingProxy {
  std::stringstream file_line;

  ScopedTimingProxy(const char *file, int line, const char *func) {
    file_line << common::get_file_from_filepath(file) << ":" << line << " "
              << func;
  }

  template <typename T> auto operator+(T &&func) {
    return ScopedTiming<std::decay_t<decltype(func)>>(
        std::forward<decltype(func)>(func), file_line.str());
  }
};
}
}

#endif // PACXX_V2_TIMING_H
