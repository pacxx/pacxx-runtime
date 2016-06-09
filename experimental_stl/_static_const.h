#pragma once

namespace pacxx {
namespace exp {
inline namespace v1 {
template <typename T> struct static_const { static constexpr T value{}; };

template <typename T> constexpr T static_const<T>::value;
}
}
}
