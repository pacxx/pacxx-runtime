#pragma once

namespace pacxx {
namespace exp {
inline namespace v1 {
template <class T> struct is_vector : std::false_type {};

template <class T, class Alloc>
struct is_vector<std::vector<T, Alloc>> : std::true_type {};

template <class T> struct is_scalar : std::false_type {};

template <class T> struct is_scalar<scalar<T>> : std::true_type {};

template <template <typename...> class Template, typename T>
struct is_specialization_of : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template, Template<Args...>> : std::true_type {};

//// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4502.pdf.
// template <typename...> using void_t = void;

//// Primary template handles all types not supporting the operation.
// template <typename, template <typename> class, typename = void_t<>>
// struct detect : std::false_type {
//};

//// Specialization recognizes/validates only types supporting the archetype.
// template <typename T, template <typename> class Op>
// struct detect<T, Op, void_t<Op<T>>> : std::true_type {
//};

//// Archetypal expression for assignment operation.
// template <typename T>
// using assign_t = decltype(std::declval<T&>() = std::declval<T const&>());

//// Trait corresponding to that archetype.
// template <typename T> using is_assignable = detect<T, assign_t>;

template <typename T> struct void_ { typedef void type; };

template <typename T, typename = void> struct is_constructable : std::false_type {};

template <typename T>
struct is_constructable<T, typename void_<typename T::construction_type>::type>
    : std::true_type {};

template <typename T, bool = is_constructable<T>::value> struct view_traits {};

template <typename T> struct view_traits<T, true> {
  using construction_type = typename T::construction_type;
  static const unsigned arity = std::tuple_size<construction_type>::value;
};

template <typename T> struct view_traits<T, false> {
  using construction_type = void;
  static const unsigned arity = 0;
};
}
}
}
