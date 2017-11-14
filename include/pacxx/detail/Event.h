//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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