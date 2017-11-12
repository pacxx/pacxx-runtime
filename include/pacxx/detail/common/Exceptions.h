//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_EXCEPTIONS_H
#define PACXX_V2_EXCEPTIONS_H
#include "Log.h"
#include <exception>
#include <string>

namespace pacxx {
namespace common {
class generic_exception : public std::exception {
public:
  generic_exception(const char *msg) : _msg(msg) { __exception(msg); }
  generic_exception(std::string msg) : _msg(msg) { __exception(msg); }
  virtual ~generic_exception() noexcept {}

  virtual const char *what() noexcept { return _msg.c_str(); }
  virtual const char *what() const noexcept { return _msg.c_str(); }

private:
  std::string _msg;
};
}
}

#endif // PACXX_V2_EXCEPTIONS_H
