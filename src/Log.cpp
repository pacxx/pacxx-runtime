//
// Created by mhaidl on 03/06/16.
//
#include "detail/common/Log.h"
namespace pacxx
{
namespace common
{

  Log& Log::get() {
    static Log the_log;
    return the_log;
  }

  Log::Log() : _silent(false), _no_warnings(false), output(std::cout) {
    _old_buffer = output.rdbuf();
    auto str = GetEnv("PACXX_LOG_LEVEL");
    log_level = 0;
    if (str.length() > 0) {
      log_level = std::stoi(str);
    }
  }

  Log::~Log() { resetStream(); }
}
}