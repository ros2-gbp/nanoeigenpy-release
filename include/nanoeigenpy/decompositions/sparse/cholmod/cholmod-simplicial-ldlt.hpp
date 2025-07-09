/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/cholmod/cholmod-decomposition.hpp"

namespace nanoeigenpy {
namespace nb = nanobind;

template <typename _MatrixType, int _UpLo = Eigen::Lower>
void exposeCholmodSimplicialLDLT(nb::module_ m, const char *name) {
  using MatrixType = _MatrixType;
  using Solver = Eigen::CholmodSimplicialLDLT<_MatrixType, _UpLo>;

  if (check_registration_alias<Solver>(m)) {
    return;
  }
  nb::class_<Solver>(
      m, name,
      "A simplicial direct Cholesky (LDLT) factorization and solver based on "
      "Cholmod.\n"
      "This class allows to solve for A.X = B sparse linear problems via a "
      "simplicial LL^T Cholesky factorization using the Cholmod library."
      "This simplicial variant is equivalent to Eigen's built-in "
      "SimplicialLDLT class."
      "Therefore, it has little practical interest. The sparse matrix A "
      "must "
      "be selfadjoint and positive definite."
      "The vectors or matrices X and B can be either dense or sparse.")

      .def(nb::init<>(), "Default constructor.")
      .def(nb::init<const MatrixType &>(), nb::arg("matrix"),
           "Constructs a LDLT factorization from a given matrix.")

      .def(CholmodBaseVisitor());
}

}  // namespace nanoeigenpy
