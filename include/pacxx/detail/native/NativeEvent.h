//
// Created by m_haid02 on 29.06.17.
//

#pragma once

#include "pacxx/detail/Event.h"
#include <chrono>

namespace pacxx{
namespace v2{

class NativeEvent : public Event{
public:
  NativeEvent() {}
  virtual ~NativeEvent() {}

  virtual void start() override;
  virtual void stop() override;

  virtual double result() override;
private:
  std::chrono::high_resolution_clock::time_point _start, _end;
};

}
}