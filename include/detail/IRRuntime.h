//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_IRRUNTIME_H
#define PACXX_V2_IRRUNTIME_H

#include <string>
#include <vector>

namespace pacxx
{
  namespace v2
  {
    class IRRuntime
    {
    public:
      virtual void linkMC(const std::string& MC) = 0;
      virtual void setArguments(std::vector<char> args) = 0;
    };
  }
}



#endif //PACXX_V2_IRRUNTIME_H
