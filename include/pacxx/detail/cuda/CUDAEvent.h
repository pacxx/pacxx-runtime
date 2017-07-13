//
// Created by m_haid02 on 29.06.17.
//

#pragma once

#include "pacxx/detail/Event.h"
#include <chrono>

// TODO: Rewrite for CUDA events

namespace pacxx{
namespace v2{

class CUDAEvent : public Event{
public:
  CUDAEvent() {}
  virtual ~CUDAEvent() {}

  virtual void start() override;
  virtual void stop() override;

  virtual double result() override;
private:
  std::chrono::high_resolution_clock::time_point _start, _end;
};

}
}