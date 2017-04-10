//
// Created by mhaidl on 8/16/16.
//

#ifndef PACXX_V2_PROMISE_H
#define PACXX_V2_PROMISE_H

#include <future>

namespace pacxx {
namespace v2 {

class Executor;

template <typename PromisedTy> class BindingPromise {
private:
  friend class Executor;

  template <typename... Ts>
  BindingPromise(Ts &&... args) : _bound(std::forward<Ts>(args)...) {}

  ~BindingPromise() {}

public:
  auto &getBoundObject() { return _bound; }

  auto getFuture() { return _promise.get_future(); }

  void fulfill() { return _promise.set_value(std::move(_bound)); }

private:
  std::promise<PromisedTy> _promise;
  PromisedTy _bound;
};
}
}
#endif // PACXX_V2_PROMISE_H
