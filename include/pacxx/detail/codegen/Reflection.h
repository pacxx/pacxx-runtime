//
// Created by m_haid02 on 19.05.17.
//

#pragma once

#include <utility>

namespace pacxx {
namespace v2 {
template<typename F, typename... Args>
[[pacxx::staging]] __attribute__((fastcall)) auto _stage(F func, Args &&... args) {
  return static_cast<long>(func(std::forward<Args>(args)...));
}

template<typename Arg> auto stage(const Arg &val) {
  return static_cast<Arg>(_stage([&] { return val; }));
}
}
}