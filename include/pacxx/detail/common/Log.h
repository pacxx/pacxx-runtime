//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_LOG_H
#define PACXX_V2_LOG_H

#ifndef PACXX_PASS_NAME
#define PACXX_PASS_NAME ""
#endif
#define __dump(a)                                                              \
  pacxx::common::dumpToLog((a), PACXX_PASS_NAME, __FILE__, __LINE__)

namespace pacxx {
namespace common {

class Log;

struct LOG_LEVEL {

  class Log;

  enum LEVEL {
    info = 0,
    debug = 1,
    verbose = 2,
    warning = -1,
    exception = -2,
    error = -3,
    fatal = -4,
    none = -999
  };
};
} // namespace common
} // namespace pacxx

template <pacxx::common::LOG_LEVEL::LEVEL debug_level =
              pacxx::common::LOG_LEVEL::info,
          typename... Params>
static void pacxx_log_print(const char *file, int line, Params &&... args);

// TODO: remove when the old Log.h is removed
#ifdef __message
#undef __message
#endif

#ifdef __debug
#undef __debug
#endif

#ifdef __warning
#undef __warning
#endif

#ifdef __error
#undef __error
#endif

#ifdef __verbose
#undef __verbose
#endif

#ifdef __fatal
#undef __fatal
#endif

#ifdef __exception
#undef __exception
#endif

#define __message(...) pacxx_log_print<>(__FILE__, __LINE__, __VA_ARGS__)
#define __warning(...)                                                         \
  pacxx_log_print<pacxx::common::LOG_LEVEL::warning>(__FILE__, __LINE__,       \
                                                     __VA_ARGS__)
#define __error(...)                                                           \
  pacxx_log_print<pacxx::common::LOG_LEVEL::error>(__FILE__, __LINE__,         \
                                                   __VA_ARGS__)
#define __debug(...)                                                           \
  pacxx_log_print<pacxx::common::LOG_LEVEL::debug>(__FILE__, __LINE__,         \
                                                   __VA_ARGS__)
#define __verbose(...)                                                         \
  pacxx_log_print<pacxx::common::LOG_LEVEL::verbose>(__FILE__, __LINE__,       \
                                                     __VA_ARGS__)
#define __fatal(...)                                                           \
  pacxx_log_print<pacxx::common::LOG_LEVEL::fatal>(__FILE__, __LINE__,         \
                                                   __VA_ARGS__)
#define __exception(...) //                                   \
  //pacxx_log_print<pacxx::common::LOG_LEVEL::exception>(__FILE__, __LINE__, __VA_ARGS__)

#include <iostream>
#include <sstream>
#include <string>

#include "Common.h"
#include "TearDown.h"

namespace llvm {
class Value;
}

namespace pacxx {

namespace common {

void dumpToLog(const llvm::Value &V, std::string prefix = "",
               const char *file = "", int line = 0);

class Log {
public:
  static Log &get();

private:
  friend void pacxx::v2::pacxxTearDown();

  Log();
  virtual ~Log();

public:
  void setStream(std::ostream &stream) { output.rdbuf(stream.rdbuf()); }

  void resetStream() { output.rdbuf(_old_buffer); }

  template <LOG_LEVEL::LEVEL debug_level = LOG_LEVEL::info, typename... Params>
  void print(const char *file, int line, Params &&... args) {

    if (_silent)
      return;
    if (log_level < debug_level)
      return;

    std::stringstream ss;
    std::stringstream file_line;
    file_line << common::get_file_from_filepath(file) << ":" << line;

    std::string program("");

    size_t label_length = 5;

    switch (debug_level) {
    case LOG_LEVEL::exception:
      ss << "EXCEPTION: ";
    case LOG_LEVEL::fatal:
    case LOG_LEVEL::error:
      label_length = 6;
      break;
    case LOG_LEVEL::warning:
      if (_no_warnings)
        return;
      label_length = 8;
      break;
    case LOG_LEVEL::info:
      break;
    case LOG_LEVEL::debug:
      ss << "DEBUG: ";
      break;
    case LOG_LEVEL::verbose:
      ss << "VERBOSE: ";
      break;
    default:
      break;
    }

    size_t offset =
        file_line.str().length() + 2 + label_length + 1 + ss.str().length();

    printValue<Params...>(ss, std::forward<decltype(args)>(args)...);

    std::string replacement(offset, ' ');

    std::string msg_str =
        common::replace_substring(ss.str(), "\n", "\n" + replacement);

    printDiagnositc(program, file_line, msg_str, debug_level);
  }

private:
  void printDiagnositc(const std::string &program,
                       const std::stringstream &file_line,
                       const std::string &msg_str,
                       LOG_LEVEL::LEVEL debug_level);

  template <typename T, typename... Params>
  void printValue(std::stringstream &ss, T &&first, Params &&... other) {
    ss << first;

    if (sizeof...(other)) {
      printValue(ss, std::forward<decltype(other)>(other)...);
    }
  }

  void printValue(std::stringstream &) {}

private:
  bool _silent;
  bool _no_warnings;
  int log_level;
  std::ostream &output;
  std::streambuf *_old_buffer;
};
} // namespace common
} // namespace pacxx

template <pacxx::common::LOG_LEVEL::LEVEL debug_level, typename... Params>
static void pacxx_log_print(const char *file, int line, Params &&... args) {
  pacxx::common::Log::get().print<debug_level, Params...>(
      file, line, std::forward<decltype(args)>(args)...);
}
#endif // PACXX_V2_LOG_H
