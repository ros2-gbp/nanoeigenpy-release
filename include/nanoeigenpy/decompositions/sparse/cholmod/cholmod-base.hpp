/// Copyright 2025 INRIA

#pragma once

#include "nanoeigenpy/fwd.hpp"
#include "nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp"
#include <Eigen/CholmodSupport>

namespace nanoeigenpy {

struct CholmodBaseVisitor : nb::def_visitor<CholmodBaseVisitor> {
  template <typename CholdmodDerived, typename... Ts>
  void execute(nb::class_<CholdmodDerived, Ts...> &cl) {
    using Solver = CholdmodDerived;
    static_assert(nb::is_base_of_template_v<Solver, Eigen::SparseSolverBase>,
                  "Template type parameter Solver must inherit from "
                  "Eigen::SparseSolverBase");
    using MatrixType = typename CholdmodDerived::MatrixType;

    cl.def("analyzePattern", &Solver::analyzePattern,
           "Performs a symbolic decomposition on the sparcity of matrix.\n"
           "This function is particularly useful when solving for several "
           "problems having the same structure.")

        .def(SparseSolverBaseVisitor())

        .def(
            "compute",
            [](Solver &self, MatrixType const &matrix) -> decltype(auto) {
              return self.compute(matrix);
            },
            nb::arg("matrix"),
            "Computes the sparse Cholesky decomposition of a given matrix.",
            nb::rv_policy::reference)

        .def("determinant", &Solver::determinant,
             "Returns the determinant of the underlying matrix from the "
             "current factorization.")

        .def("factorize", &Solver::factorize, nb::arg("matrix"),
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.\n"
             "See also analyzePattern().")

        .def("info", &Solver::info,
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("logDeterminant", &Solver::logDeterminant,
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("setShift", &Solver::setShift, nb::arg("offset"),
             "Sets the shift parameters that will be used to adjust the "
             "diagonal coefficients during the numerical factorization.\n"
             "During the numerical factorization, the diagonal coefficients "
             "are transformed by the following linear model: d_ii = offset + "
             "d_ii.\n"
             "The default is the identity transformation with offset=0.",
             nb::rv_policy::reference);
  }
};

}  // namespace nanoeigenpy
