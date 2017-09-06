//
// Created by m_haid02 on 06.09.17.
//

#include "pacxx/Executor.h"
#include "pacxx/detail/common/Common.h"

#include <llvm/Support/raw_ostream.h>


namespace pacxx{
namespace v2 {

bool __isTearingDown = false;

void __attribute__ ((destructor)) pacxxTearDown(){

  __isTearingDown = true;

  auto &executors = Executor::getExecutors();
  try {
    //executors.clear();
  }
  catch(...){ } // ignore exceptions in this stage

  //delete &executors;
}
}
}