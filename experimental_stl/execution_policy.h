#pragma once
#include "detail/KernelConfiguration.h"

namespace pacxx
{
    namespace exp
    {
        class pacxx_execution_policy
        {
          public:
            pacxx_execution_policy(pacxx::v2::KernelConfiguration config)
                : config(config)
            {
            }
            virtual ~pacxx_execution_policy() {}

            auto getConfig() { return config; }
          private:
            pacxx::v2::KernelConfiguration config;
        };
    }
}
