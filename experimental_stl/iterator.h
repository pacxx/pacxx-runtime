#pragma once

#include <type_traits>
namespace pacxx {
namespace exp {
template <typename T> class scalar final {
public:
  using reference = std::remove_reference_t<T> &;
  using const_reference = const std::remove_reference_t<T> &;

  scalar(T value) : value(value) {}
  scalar(T &&value) : value(value) {}

  reference operator[](int) { return value; }
  const_reference operator[](int) const { return value; }

private:
  T value;
};

template <class T> struct is_container : std::false_type {};

template <class T, class Alloc>
struct is_container<std::vector<T, Alloc>> : std::true_type {};

template <class T> struct is_scalar : std::false_type {};

template <class T> struct is_scalar<scalar<T>> : std::true_type {};

template <typename _Type> struct arg_wrapper {
  using type = std::conditional_t<
      !is_container<std::remove_reference_t<_Type>>::value,
      scalar<std::remove_reference_t<_Type>> &, _Type>;
};

template <typename... Ts> struct arg_iterator {
  arg_iterator(Ts &&... args) : _args(std::tuple<Ts...>(args...)) {
    _size = std::get<0>(_args).size();
  };
  auto operator*() {
    return apply([](auto &&... args) {
      return get_args_as_tuple((*std::begin(args))...);
    }, _args);
  }

  std::tuple<Ts...> _args;
  size_t _size;
  size_t size() const { return _size; }
};

template <typename... Ts> auto make_iterator(Ts &&... args) {
  return arg_iterator<typename arg_wrapper<Ts>::type...>(
      typename arg_wrapper<Ts>::type(args)...);
}
}
}
