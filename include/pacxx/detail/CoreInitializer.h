//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_COREINITIALIZER_H
#define PACXX_V2_COREINITIALIZER_H

namespace pacxx {
namespace core {
class CoreInitializer {
public:
  static void initialize();

private:
  CoreInitializer();
  ~CoreInitializer();
  void initializeCore();
  bool _initialized;
};
}
}

#endif // PACXX_V2_COREINITIALIZER_H
