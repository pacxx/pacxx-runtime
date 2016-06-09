#pragma once
#include <cstddef>
#include <tuple>
#include "tuple.h"
#include <type_traits>
#include "./type_traits.h"
#include "_static_const.h"
#include "range/v3/all.hpp"

namespace __ranges = ranges;

namespace pacxx {
namespace exp {
inline namespace v1 {
template <typename T> class vector_view {
public:
  using RngTy = vector_view<T>;

  using base_type = T;

  using reference = typename T::reference;
  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using difference_type = std::ptrdiff_t;

  struct iterator : public __ranges::random_access_iterator_tag {
    using base_type = T;
    using range = vector_view<base_type>;
    using difference_type = std::ptrdiff_t;
    using value_type = typename base_type::value_type;
    using reference = value_type &;
    using rvalue_reference = value_type &&;
    using iterator_category = __ranges::random_access_iterator_tag;

    iterator() = default;
    iterator(typename T::iterator it, T *__owner) : it(it), __owner(__owner) {}

    reference operator*() const { return *it; }

    iterator &operator++() {
      ++it;
      return *this;
    }

    iterator operator++(int) {
      auto ip = it;
      ++it;
      return iterator(ip, __owner);
    }

    iterator &operator--() {
      --it;
      return *this;
    }

    iterator operator--(int) {
      auto ip = it;
      --it;
      return iterator(ip, __owner);
    }

    reference operator[](difference_type n) { return *(it + n); }

    friend iterator operator+(const iterator &lhs, difference_type n) {
      return iterator(lhs.it + n, lhs.__owner);
    }
    friend iterator operator+(difference_type n, const iterator &rhs) {
      return iterator(rhs.it + n, rhs.__owner);
    }
    friend iterator operator-(const iterator &lhs, difference_type n) {
      return iterator(lhs.it - n, lhs.__owner);
    }

    friend difference_type operator-(const iterator &left,
                                     const iterator &right) {
      return left.it - right.it;
    }

    friend iterator &operator-=(iterator &lhs, difference_type n) {
      lhs.it -= n;
      return lhs;
    }
    friend iterator &operator+=(iterator &lhs, difference_type n) {
      lhs.it -= n;
      return lhs;
    }

    friend bool operator<(const iterator &left, const iterator &right) {
      return left.it < right.it;
    }

    friend bool operator>(const iterator &left, const iterator &right) {
      return left.it > right.it;
    }

    friend bool operator<=(const iterator &left, const iterator &right) {
      return left.it <= right.it;
    }

    friend bool operator>=(const iterator &left, const iterator &right) {
      return left.it >= right.it;
    }

    bool operator==(const iterator &other) { return other.it == it; }

    bool operator!=(const iterator &other) { return other.it != it; }

    T &owner() { return *__owner; }

  private:
    typename T::iterator it;
    T *__owner;
  };

  using sentinel = iterator;
  using construction_type = std::tuple<T &>;
  vector_view() = default;
  vector_view(T &vec) : __owner(&vec) {}
  vector_view(construction_type &t) : __owner(&std::get<0>(t)) {}

  iterator begin() { return iterator(std::begin(*__owner), __owner); }
  sentinel end() { return sentinel(std::end(*__owner), __owner); }

private:
  T *__owner;
};

template <typename T> class scalar_view {
public:
  using RngTy = scalar_view<T>;

  using base_type =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;

  using reference = base_type &;
  using value_type = T;
  using size_type = typename base_type::size_type;
  using difference_type = std::ptrdiff_t;

  struct iterator : public __ranges::input_iterator_tag {
    using base_type =
        typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    using range = scalar_view<T>;
    using difference_type = std::ptrdiff_t;
    using value_type = base_type;
    using reference = const value_type &;
    using rvalue_reference = value_type &&;
    using iterator_category = __ranges::input_iterator_tag;

    iterator() = default;
    iterator(const base_type *__owner) : __owner(__owner) {}

    reference operator*() { return *__owner; }
    value_type operator*() const { return *__owner; }

    iterator &operator++() { return *this; }

    iterator operator++(int) { return iterator(__owner); }

    iterator &operator--() { return *this; }

    iterator operator--(int) { return iterator(__owner); }

    reference operator[](difference_type) { return *__owner; }

    friend iterator operator+(const iterator &lhs, difference_type) {
      return iterator(lhs.__owner);
    }
    friend iterator operator+(difference_type, const iterator &rhs) {
      return iterator(rhs.__owner);
    }
    friend iterator operator-(const iterator &lhs, difference_type) {
      return iterator(lhs.__owner);
    }

    friend difference_type operator-(const iterator &, const iterator &) {
      return difference_type();
    }

    friend iterator &operator-=(iterator &lhs, difference_type) { return lhs; }
    friend iterator &operator+=(iterator &lhs, difference_type) { return lhs; }

    friend bool operator<(const iterator &, const iterator &) { return false; }

    friend bool operator>(const iterator &, const iterator &) { return false; }

    friend bool operator<=(const iterator &, const iterator &) { return false; }

    friend bool operator>=(const iterator &, const iterator &) { return false; }

    bool operator==(const iterator &other) { return other.__owner == __owner; }

    bool operator!=(const iterator &other) { return other.__owner != __owner; }

    reference owner() { return *__owner; }

    auto unwrap() { return construction_type(*__owner); }

  private:
    const base_type *__owner;
  };

  using sentinel = iterator;

  using construction_type = std::tuple<const base_type &>;

  scalar_view() = default;
  scalar_view(base_type &vec) : __owner(&vec) {}
  scalar_view(construction_type tuple) : __owner(&std::get<0>(tuple)) {}

  static scalar_view Create(base_type &vec) { return scalar_view(vec); }

  static scalar_view CreateTP(construction_type &tuple) {
    return scalar_view(tuple);
  }

  iterator begin() { return iterator(__owner); }
  sentinel end() { return sentinel(__owner); }

private:
  const base_type *__owner;
};

template <typename T> class value_view {
public:
  using RngTy = value_view<T>;

  using base_type = T;

  using reference = base_type &;
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  struct iterator : public __ranges::input_iterator_tag {
    using base_type = T;
    using range = value_view<T>;
    using difference_type = std::ptrdiff_t;
    using value_type = base_type;
    using reference = value_type &;
    using rvalue_reference = value_type &&;
    using iterator_category = __ranges::input_iterator_tag;

    iterator() = default;
    iterator(const base_type __owner) : __owner(__owner) {}

    reference operator*() { return __owner; }
    value_type operator*() const { return *__owner; }

    iterator &operator++() { return *this; }

    iterator operator++(int) { return iterator(__owner); }

    iterator &operator--() { return *this; }

    iterator operator--(int) { return iterator(__owner); }

    reference operator[](difference_type) { return __owner; }

    friend iterator operator+(const iterator &lhs, difference_type) {
      return iterator(lhs.__owner);
    }
    friend iterator operator+(difference_type, const iterator &rhs) {
      return iterator(rhs.__owner);
    }
    friend iterator operator-(const iterator &lhs, difference_type) {
      return iterator(lhs.__owner);
    }

    friend difference_type operator-(const iterator &, const iterator &) {
      return difference_type();
    }

    friend iterator &operator-=(iterator &lhs, difference_type) { return lhs; }
    friend iterator &operator+=(iterator &lhs, difference_type) { return lhs; }

    friend bool operator<(const iterator &, const iterator &) { return false; }

    friend bool operator>(const iterator &, const iterator &) { return false; }

    friend bool operator<=(const iterator &, const iterator &) { return false; }

    friend bool operator>=(const iterator &, const iterator &) { return false; }

    bool operator==(const iterator &other) { return other.__owner == __owner; }

    bool operator!=(const iterator &other) { return other.__owner != __owner; }

    reference owner() { return __owner; }

    auto unwrap() { return construction_type(__owner); }

  private:
    base_type __owner;
  };

  using sentinel = iterator;

  using construction_type = std::tuple<const base_type>;

  value_view() = default;
  value_view(base_type val) : __owner(val) {}
  value_view(construction_type tuple) : __owner(std::get<0>(tuple)) {}

  static value_view Create(base_type &vec) { return value_view(vec); }

  static value_view CreateTP(construction_type &tuple) {
    return value_view(tuple);
  }

  iterator begin() { return iterator(__owner); }
  sentinel end() { return sentinel(__owner); }

private:
  base_type __owner;
};


template <typename T, size_t width> class __matrix {
public:
  using RngTy = __matrix<T, width>;

  using base_type = T;

  using construction_type = std::tuple<T &>;

  using reference = typename T::reference;
  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using difference_type = std::ptrdiff_t;

  struct iterator : public __ranges::random_access_iterator_tag {
    using base_type = T;
    using range = __matrix<base_type, width>;
    using difference_type = std::ptrdiff_t;
    using value_type = typename base_type::value_type;
    using reference = value_type &;
    using rvalue_reference = value_type &&;
    using iterator_category = __ranges::random_access_iterator_tag;

    iterator() = default;
    iterator(typename T::iterator it, T *__owner) : it(it), __owner(__owner) {}

    reference operator*() { return *it; }
    value_type operator*() const { return *it; }

    iterator &operator++() {
      ++it;
      return *this;
    }

    iterator operator++(int) {
      auto ip = it;
      ++it;
      return iterator(ip, __owner);
    }

    iterator &operator--() {
      --it;
      return *this;
    }

    iterator operator--(int) {
      auto ip = it;
      --it;
      return iterator(ip, __owner);
    }

    reference operator[](std::pair<difference_type, difference_type> n) {
      return *(it + n.second * width + n.first);
    }

    friend iterator operator+(const iterator &lhs,
                              std::pair<difference_type, difference_type> n) {
      return iterator(lhs.it + n.second * width + n.first, lhs.__owner);
    }
    friend iterator operator+(std::pair<difference_type, difference_type> n,
                              const iterator &rhs) {
      return iterator(rhs.it + n.second * width + n.first, rhs.__owner);
    }
    friend iterator operator-(const iterator &lhs,
                              std::pair<difference_type, difference_type> n) {
      return iterator(lhs.it - n.second * width + n.first, lhs.__owner);
    }

    friend difference_type operator-(const iterator &left,
                                     const iterator &right) {
      return left.it - right.it;
    }

    friend iterator &operator-=(iterator &lhs,
                                std::pair<difference_type, difference_type> n) {
      lhs.it -= n.second * width + n.first;
      return lhs;
    }
    friend iterator &operator+=(iterator &lhs,
                                std::pair<difference_type, difference_type> n) {
      lhs.it -= n.second * width + n.first;
      return lhs;
    }

    friend bool operator<(const iterator &left, const iterator &right) {
      return left.it < right.it;
    }

    friend bool operator>(const iterator &left, const iterator &right) {
      return left.it > right.it;
    }

    friend bool operator<=(const iterator &left, const iterator &right) {
      return left.it <= right.it;
    }

    friend bool operator>=(const iterator &left, const iterator &right) {
      return left.it >= right.it;
    }

    bool operator==(const iterator &other) { return other.it == it; }

    bool operator!=(const iterator &other) { return other.it != it; }

    T &owner() { return *__owner; }
    auto unwrap() { return construction_type(*__owner); }

  private:
    typename T::iterator it;
    T *__owner;
  };

  using sentinel = iterator;

  __matrix() = default;
  __matrix(construction_type tuple) : __owner(&std::get<0>(tuple)) {}

  T &owner() { return *__owner; }

  iterator begin() { return iterator(std::begin(*__owner), __owner); }
  sentinel end() { return sentinel(std::end(*__owner), __owner); }

  size_t size() { return __owner->size(); }

private:
  T *__owner;
};

template <typename T, template <typename> class STENCIL,
          typename VALUES = typename T::value_type>
class __stencil {
public:
  using RngTy = __stencil<T, STENCIL, VALUES>;

  using base_type = T;

  using reference = typename T::reference;
  using size_type = typename T::size_type;
  using value_type = typename T::value_type;
  using difference_type = std::ptrdiff_t;

  using construction_type =
      std::tuple<const T &, unsigned int, unsigned int, unsigned int>;

  struct iterator : public __ranges::random_access_iterator_tag {
    using base_type = T;
    using difference_type = std::ptrdiff_t;
    using value_type = STENCIL<const VALUES>;
    using range = RngTy;
    using reference = const value_type &;
    using rvalue_reference = value_type &&;
    using iterator_category = __ranges::random_access_iterator_tag;

    iterator() = default;
    iterator(const iterator &) = default;
    iterator(iterator &&) = default;
    iterator &operator=(const iterator &) = default;
    iterator &operator=(iterator &&) = default;

    iterator(const T *__owner, unsigned int width, unsigned int height,
             unsigned int _range, unsigned int x = 0, unsigned int y = 0)
        : __owner(__owner),
          V(_range, (typename value_type::data_type) & (*__owner)[0], width,
            height, x, y),
          width(width), height(height), _range(_range), x(x), y(y) {}

    reference operator*() const { return V; }

    iterator &operator++() {
      ++x;
      if (x >= width) {
        ++y;
        x = 0;
      }
      return *this;
    }

    iterator operator++(int) {
      auto ox = x;
      auto oy = y;

      ++x;
      if (x >= width) {
        ++y;
        x = 0;
      }
      return iterator(__owner, width, height, _range, ox, oy);
    }

    iterator &operator--() {
      --x;
      if (x < 0) {
        --y;
        x = width - 1;
      }
      return *this;
    }

    iterator operator--(int) {
      auto ox = x;
      auto oy = y;
      --x;
      if (x < 0) {
        --y;
        x = width - 1;
      }
      return iterator(__owner, width, height, _range, ox, oy);
    }

    value_type operator[](std::pair<difference_type, difference_type> n) {
      return value_type(_range, &(*__owner)[0], width, height, x + n.first,
                        y + n.second);
    }

    friend iterator operator+(const iterator &lhs,
                              std::pair<difference_type, difference_type> n) {
      return iterator(lhs.__owner, lhs.width, lhs.height, lhs._range,
                      lhs.x + n.first, lhs.y + n.second);
    }
    friend iterator operator+(std::pair<difference_type, difference_type> n,
                              const iterator &rhs) {
      return iterator(rhs.__owner, rhs.width, rhs.height, rhs._range,
                      rhs.x + n.first, rhs.y + n.second);
    }
    friend iterator operator-(const iterator &lhs,
                              std::pair<difference_type, difference_type> n) {
      return iterator(lhs.__owner, lhs.width, lhs.height, lhs._range,
                      lhs.x - n.first, lhs.y - n.second);
    }

    friend auto operator-(const iterator &left, const iterator &right) {
      return std::pair<difference_type, difference_type>(left.x - right.x,
                                                         left.y - right.y);
    }

    friend iterator &operator-=(iterator &lhs,
                                std::pair<difference_type, difference_type> n) {
      lhs.x -= n.first;
      lhs.y -= n.second;
      return lhs;
    }
    friend iterator &operator+=(iterator &lhs,
                                std::pair<difference_type, difference_type> n) {
      lhs.x += n.first;
      lhs.y += n.second;
      return lhs;
    }

    friend bool operator<(const iterator &left, const iterator &right) {
      return left.x < right.x && left.y < right.y;
    }

    friend bool operator>(const iterator &left, const iterator &right) {
      return left.x > right.x && left.y > right.y;
    }

    friend bool operator<=(const iterator &left, const iterator &right) {
      return left.x <= right.x && left.y <= right.y;
    }

    friend bool operator>=(const iterator &left, const iterator &right) {
      return left.x >= right.x && left.y >= right.y;
    }

    bool operator==(const iterator &other) {
      return other.x == x && other.y == y;
    }

    bool operator!=(const iterator &other) {
      return other.x != x || other.y != y;
    }

    const T &owner() { return *__owner; }
    auto unwrap() { return construction_type(*__owner, width, height, _range); }

  private:
    const T *__owner;
    value_type V;
    unsigned int width, height, _range;
    unsigned int x, y;
  };

  using sentinel = iterator;

  //__stencil() = default;

  static __stencil Create(const T &vec, unsigned int width, unsigned int height,
                          unsigned int range) {
    return __stencil(construction_type(vec, width, height, range));
  }

  static __stencil CreateTP(construction_type &tuple) {
    return __stencil(tuple);
  }

  __stencil(construction_type tuple)
      : __owner(&std::get<0>(tuple)), width(std::get<1>(tuple)),
        height(std::get<2>(tuple)), range(std::get<3>(tuple)) {}

  const T &owner() { return *__owner; }

  iterator begin() { return iterator(__owner, width, height, range); }
  sentinel end() { return sentinel(__owner, width, height, range); }

  size_t size() { return __owner->size(); }

private:
  const T *__owner;
  unsigned int width, height, range;
};

template <size_t width, typename T> auto matrix(T &vec) {
  return __matrix<T, width>(vec);
}

template <template <typename> class STENCIL, typename VALUES = void, typename T>
auto stencil(T &vec, unsigned int width, unsigned int height,
             unsigned int range) {
  using value_type = std::conditional_t<std::is_void<VALUES>::value,
                                        typename T::value_type, VALUES>;
  return __stencil<T, STENCIL, value_type>(
      typename __stencil<T, STENCIL, value_type>::construction_type(
          vec, width, height, range));
}
}
}

namespace view {
template <typename T,
          typename std::enable_if<exp::is_vector<T>::value>::type * = nullptr>
auto __decorate(T &&vec) {
  return vec; // exp::vector_view<T>(vec);
}

template <typename T,
          typename std::enable_if<!exp::is_vector<T>::value>::type * = nullptr>
auto __decorate(T &&vec) {
  return vec;
}

struct __zip {
  template <typename... Rng> auto operator()(Rng &... rngs) const {
    return __ranges::view::zip(__decorate(std::forward<Rng>(rngs))...);
  }
};

constexpr auto &&zip = exp::static_const<__zip>::value;
}
}
