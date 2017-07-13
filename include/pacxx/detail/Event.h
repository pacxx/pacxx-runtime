//
// Created by m_haid02 on 29.06.17.
//

#pragma once

namespace pacxx{
namespace v2{
  class Event
  {
  public:
    virtual ~Event() {}

    virtual void start() = 0;
    virtual void stop () = 0;
    virtual double result () = 0;
  };
}
}