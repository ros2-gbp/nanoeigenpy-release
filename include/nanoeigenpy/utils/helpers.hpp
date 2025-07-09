/// Copyright 2025 INRIA

#pragma once

#include <nanobind/nanobind.h>

namespace nanoeigenpy {
namespace nb = nanobind;

/*! * Symlink to the current scope the already registered class T.
 *
 * @tparam T The class type to be symlinked.
 * @param m nanobind module.
 *
 * \returns true if the type T is effectively symlinked.
 */
template <typename T>
inline bool check_registration_alias(nb::module_& m) {
  if (nb::handle py_type = nb::type<T>(); py_type.is_valid()) {
    m.attr(py_type.attr("__name__")) = py_type;
    return true;
  }
  return false;
}

/*! * Deprecated name of the function inspired by
 * eigenpy::register_symbolic_link_to_registered_type.
 *
 * @tparam T The class type to be symlinked.
 * @param m nanobind module.
 *
 * \returns true if the type T is effectively symlinked.
 */
template <typename T>
[[deprecated("Use check_registration_alias<T>(m) instead.")]]
inline bool register_symbolic_link_to_registered_type(nb::module_& m) {
  return check_registration_alias<T>(m);
}

}  // namespace nanoeigenpy
