//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "pacxx/detail/Event.h"
#include <chrono>

// TODO: Rewrite for Remote events

namespace pacxx{
namespace v2{

class RemoteEvent : public Event{
public:
  RemoteEvent() {}
  virtual ~RemoteEvent() {}

  virtual void start() override;
  virtual void stop() override;

  virtual double result() override;
private:
  std::chrono::high_resolution_clock::time_point _start, _end;
};

}
}