#pragma once
#include <utility>
#include <tuple>
#include "range/v3/utility/common_tuple.hpp"

namespace pacxx
{
    namespace exp
    {
        inline namespace v1
        {
            template <size_t N> struct Apply {
                template <typename F, typename T, typename... A>
                static inline auto apply(F&& f, T&& t, A&&... a)
                {
                    return Apply<N - 1>::apply(::std::forward<F>(f), ::std::forward<T>(t),
                                               ::std::get<N - 1>(::std::forward<T>(t)),
                                               ::std::forward<A>(a)...);
                }
            };

            template <> struct Apply<0> {
                template <typename F, typename T, typename... A>
                static inline auto apply(F&& f, T&&, A&&... a)
                {
                    return ::std::forward<F>(f)(::std::forward<A>(a)...);
                }
            };
        }

        template <typename F, typename T> inline auto apply(F&& f, T&& t)
        {
            return Apply<::std::tuple_size<::std::decay_t<T>>::value>::apply(
                ::std::forward<F>(f), ::std::forward<T>(t));
        }


        template <typename F, typename... Ts>
        inline auto apply(F&& f, ranges::v3::common_tuple<Ts...>& t)
        {
            std::tuple<Ts...> std_tuple = t;
            return apply(f, std_tuple);
        }
    }
}
