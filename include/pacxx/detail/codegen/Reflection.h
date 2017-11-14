//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <utility>

namespace pacxx {
namespace v2 {
template<typename F, typename... Args>
[[pacxx::staging]] auto _stage(F func, Args &&... args) {
  return static_cast<long>(func(std::forward<Args>(args)...));
}

template<typename Arg> auto stage(const Arg &val) {
  return static_cast<Arg>(_stage([&] { return val; }));
}
}
}