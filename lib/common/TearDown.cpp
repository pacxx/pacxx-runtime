//
// Created by m_haid02 on 06.09.17.
//

#include "pacxx/Executor.h"
#include "pacxx/detail/common/Common.h"

namespace pacxx{
namespace v2 {

void pacxxTearDown(){
  auto &executors = Executor::getExecutors();
  delete &executors;
}
}
}
