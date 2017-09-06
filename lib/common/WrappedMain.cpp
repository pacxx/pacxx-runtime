//
// Created by m_haid02 on 06.09.17.
//

#include "pacxx/detail/common/TearDown.h"

extern "C" int __real_main(int argc, char* argv[]);

extern "C" int __wrap_main(int argc, char *argv[]) {
  int ret = __real_main(argc, argv);
  pacxx::v2::pacxxTearDown();
  return ret;
}