#pragma once
#ifndef pacxx_kernel
#define pacxx_kernel kernel
#endif

#include "execution_policy.h"
#include "tuple.h"
#include "iterator.h"
#include "views.h"
#include <type_traits>
#include <tuple>
#include "_static_const.h"

#include "PACXX.h"

namespace pacxx {
namespace exp {

template <typename F, typename T, typename... Ts>
auto __first(F &&func, T &&arg, Ts &&...) {
  return func(arg);
}

namespace detail {
template <typename T> struct __grid_point {
  __grid_point(T x, T y, T z) : x(x), y(y), z(z){};
  operator int() { return static_cast<int>(x); }
  operator std::ptrdiff_t() { return static_cast<long>(x); }
  operator std::pair<T, T>() { return {x, y}; }
  operator std::pair<std::ptrdiff_t, std::ptrdiff_t>() { return {x, y}; }
  T x, y, z;
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"
template <typename T,
          typename std::enable_if<!std::is_reference<
              typename T::range::value_type>::value>::type * = nullptr>
typename T::reference saveDeRef(T &&iterator, bool enabled) {
  if (enabled)
    return *iterator;
  else {
    typename T::value_type dis;
    return dis;
  }
}

template <typename T,
          typename std::enable_if<std::is_reference<
              typename T::range::value_type>::value>::type * = nullptr>
typename T::reference saveDeRef(T &&iterator, bool) {
  return *iterator;
}
#pragma clang diagnostic pop
template <typename T, typename U,
          typename std::enable_if<exp::is_vector<T>::value>::type * = nullptr>
auto __decorate(U &&vec) {
  return exp::vector_view<T>(vec);
}

template <typename T, typename U,
          typename std::enable_if<!exp::is_vector<T>::value>::type * = nullptr>
auto __decorate(U &&args) {
  return T(args);
}
}

template <typename ExecutionPolicy, template <typename... Args> class Rng,
          typename... Args, typename UnaryOp,
          ::std::enable_if_t<::std::is_same<
              typename ::std::remove_reference<ExecutionPolicy>::type,
              pacxx_execution_policy>::value> * = nullptr>
void for_each(ExecutionPolicy &&exec, Rng<Args...> &rng, UnaryOp &&func) {
  auto arg_tuple = apply([](auto &&... args) {
    return std::make_tuple(args.begin().owner()...);
  }, rng._getRngs());

  auto k = kernel([&func](typename Args::iterator::base_type &... args) {
    auto g = Thread::get().global;
    detail::__grid_point<decltype(g.x)> __id(g.x, g.y, g.z);
    /*     bool disabled = (g.x >= static_cast<int>(_stage([&] {
                              return __first([](auto&& arg) { return arg.size();
       },
                                             std::forward<decltype(args)>(args)...);
                          })));*/
    bool disabled = false;

    auto proj = [&](auto &&... args) {
      return std::forward_as_tuple(
          detail::saveDeRef(args.begin() + __id, !disabled)...);
    };

    func(proj(detail::__decorate<typename Args::iterator::range>(args)...));

  }, exec.getConfig());

  apply(k, arg_tuple);
  //    k.synchronize();
}

template <typename Rng, typename L> auto for_each(Rng &rng, L func) {
  auto size = /*_stage([&] { return */ rng.size(); /* });*/
  v2::shared_memory<typename Rng::value_type, 128> cache;
  auto block = Block::get();
  auto thread = Thread::get();

  int trips = (int)size / block.range.x;
  for (int i = 0; i < trips; ++i) {
    int offset = i * block.range.x + thread.index.x;

    cache[thread.index.x] = rng[offset];

    block.synchronize();

    for (int j = 0; j < block.range.x; ++j)
      func(cache[j]);

    block.synchronize();
  }
  if (size % block.range.x != 0)
    for (size_t j = trips * block.range.x; j < size; ++j)
      func(rng[j]);
  return func;
}

template <typename CType, typename... T, std::size_t... I>
auto materialize_(const std::tuple<T...> &t, std::index_sequence<I...>) {
  return CType(std::get<I>(t)...);
}

template <typename CType, typename... T>
auto materialize(const std::tuple<T...> &t) {
  return materialize_<CType>(t, std::make_index_sequence<sizeof...(T)>());
}

template <int Start, typename... T, std::size_t... I>
auto subtuple_(const std::tuple<T...> &t, std::index_sequence<I...>) {
  return std::tie(std::get<Start + I>(t)...);
}

template <int Start, int Length, typename... T>
auto subtuple(const std::tuple<T...> &t) {
  return subtuple_<Start>(t, std::make_index_sequence<Length>());
}

template <typename T>
struct sizeofArgs : public sizeofArgs<decltype(&T::Create)> {};

template <typename RType, typename... ArgTys>
struct sizeofArgs<RType (*)(ArgTys...)> {
  enum { arity = sizeof...(ArgTys) };
};

template <size_t length, size_t pos>
static constexpr size_t scan(const size_t arr[length], const size_t i = 0) {
  return (pos < length)
             ? ((i < pos) ? arr[i] + scan<length, pos>(arr, i + 1) : 0)
             : 0;
}

template <size_t length, size_t pos>
static constexpr size_t getAt(const size_t arr[length]) {
  return (pos < length) ? arr[pos] : 0;
}

template <typename T> struct cond_decay {
  using type =
      std::conditional_t<exp::is_vector<std::remove_reference_t<T>>::value, T,
                         std::decay_t<T>>;
};

template <typename ExecutionPolicy, typename OutputRng, typename... Args>
struct transform_invoker {
  template <typename Func, typename... OArgs, typename... KArgs, size_t... I>
  auto operator()(ExecutionPolicy &&exec, Func &func, std::tuple<OArgs...> &out,
                  std::tuple<KArgs...> &args, std::index_sequence<I...>) {

    auto arg_tuple = std::tuple_cat(out, args);

    auto k = pacxx_kernel([func](OArgs... out, const KArgs... args) {
      auto g = Thread::get().global;
      detail::__grid_point<decltype(g.x)> __id(g.x, g.y, g.z);
      /*   bool disabled = false;  (g.x >= static_cast<int>(_stage([&] {
                     return __first([](auto&& arg) { return arg.size(); },
                                    std::forward<decltype(args)>(args)...);
                 }))); */

      auto outIt = detail::__decorate<typename OutputRng::iterator::range>(
                       std::forward_as_tuple(out...))
                       .begin() +
                   __id;

      constexpr size_t a[sizeof...(Args)] = {view_traits<
          std::remove_reference_t<typename Args::iterator::range>>::arity...};

      auto targs = std::forward_as_tuple(args...);

      auto truncated =
          [&](const auto &... tuples) {
            return subtuple<0, sizeof...(Args)>(std::forward_as_tuple(tuples...));
          }(subtuple<scan<sizeof...(Args), I>(a), getAt<sizeof...(Args), I>(a)>(
              targs)...);

      auto proj = [&](auto &&... args) {
        return std::forward_as_tuple((*(args.begin() + __id))...);
        ;
        // detail::saveDeRef(args.begin() + __id, !disabled)...);
      };

      *outIt = apply([&](auto &&... args) {
        auto rngs = std::make_tuple(
            (detail::__decorate<typename Args::iterator::range>(args).begin() +
             __id)...);
        return apply([&](auto &&... args) {
          return func(std::forward_as_tuple(*args...));
          ;
        }, rngs);
      }, truncated);
    }, exec.getConfig());

    /*auto k =
                   [&func](Out out, const KArgs... args) {

                   detail::__grid_point<int> __id(0, 0, 0);
                   bool disabled = (g.x >= static_cast<int>(_stage([&] {
                                                              return
__first([](auto&&
   arg) {
   return arg.size(); },
                                                              std::forward<decltype(args)>(args)...);
                                                              })));

    auto proj = [&](auto&&... args) {
        auto t = forward_as_tuple((*(args.begin() + __id))...);

        printf("proj 0x%08x %d, %d, %d, %f\n", get<0>(t).data, get<0>(t).x,
get<0>(t).y,
               get<0>(t).range, get<1>(t)[0]);
        return t;
        // detail::saveDeRef(args.begin() + __id, !disabled)...);
    };

    auto outIt = out.begin() + __id;

    constexpr size_t a[sizeof...(Args)] = {
        view_traits<remove_reference_t<typename
Args::iterator::range>>::arity...};

    auto targs = forward_as_tuple(args...);

    auto truncated = [&](const auto&... tuples) {
        return subtuple<0, sizeof...(Args)>(forward_as_tuple(tuples...));

    }(subtuple<scan<sizeof...(Args), I>(a), getAt<sizeof...(Args),
I>(a)>(targs)...);

    *outIt = apply(
        [&](auto&&... args) {
            auto out =
                func(proj(detail::__decorate<typename
Args::iterator::range>(args)...));

            printf("got: %d\n", out);
            return out;
        },
        truncated);

    printf("outIt %d %d\n", out.size(), *outIt);
};
*/

    apply(k, arg_tuple);
  }
};

template <typename ExecutionPolicy, template <typename... Args> class InputRng,
          typename OutputRng, typename... Args, typename UnaryOp,
          ::std::enable_if_t<::std::is_same<
              typename ::std::remove_reference<ExecutionPolicy>::type,
              pacxx_execution_policy>::value> * = nullptr>
auto transform(ExecutionPolicy &&exec, InputRng<Args...> &inRng,
               OutputRng &outRng, UnaryOp &&func) {
  auto input = apply([](auto &&... args) {
    return tuple_cat(args.begin().unwrap()...);
  }, inRng._getRngs());
  auto output = outRng.begin().unwrap();

  transform_invoker<ExecutionPolicy, OutputRng, Args...> invoke;

  invoke(exec, func, output, input,
         std::make_index_sequence<std::tuple_size<decltype(input)>::value>{});
}
}
}
