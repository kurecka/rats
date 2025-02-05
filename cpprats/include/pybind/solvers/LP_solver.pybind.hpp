#pragma once
#include "solvers/LP_solver.hpp"
#include <iostream>
#include "pybind/pybind.hpp"

namespace rats::py {
    template <typename S, typename A>
    void register_lp_solver(py::module& m) {
        py::class_<LP_solver<S, A>>(m, "LP_solver")
            .def(py::init<environment<S, A>&, float>())
            .def("solve", &LP_solver<S, A>::solve)
            .def("change_thd", &LP_solver<S,A>::change_thd)
            .def("change_gammas", &LP_solver<S,A>::change_gammas, py::arg("gamma"), py::arg("gammap") = 1.f)
            .def("change_env", &LP_solver<S,A>::change_env);
    }

    void register_lp_solvers(py::module& m) {
        register_lp_solver<std::pair<int,uint64_t>, size_t>(m);
    }
} // namespace rats
