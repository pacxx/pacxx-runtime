//
// Created by mhaidl on 16/08/16.
//

#ifndef PACXX_V2_CALLBACK_H
#define PACXX_V2_CALLBACK_H

#include <detail/IRRuntime.h>
#include <detail/common/Log.h>
#include <future>

namespace pacxx {
  namespace v2 {

    class CallbackBase {
    public:
      virtual ~CallbackBase() {};

      virtual void call() = 0;

    };

    template<typename F>
    class Callback : public CallbackBase {
    public:
      Callback(F func) : _func(func), _runtime(nullptr) {}

      Callback(const Callback& other) : _func(other._func), _runtime(other._runtime) {}

      void registeredWith(IRRuntimeBase* rt) {
        _runtime = rt;
      }

      virtual void call() override {
        _func();
        _runtime->removeCallback(this);
      }

    private:
      F _func;
      IRRuntimeBase* _runtime;
    };

    template<typename CallbackFunc>
    auto make_callback(CallbackFunc&& cb) {
      return Callback<CallbackFunc>(std::forward<decltype(cb)>(cb));
    }
  }
}

#endif //PACXX_V2_CALLBACK_H
