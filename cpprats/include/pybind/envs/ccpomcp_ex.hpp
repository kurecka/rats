#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/ccpomcp_ex1.hpp"
#include "envs/ccpomcp_ex2.hpp"


namespace rats::py {

template <typename T>
void register_ccpomcp_ex(py::module &m, const T& env_type) {
    py::class_<ccpomcp_ex1>(m, "CCPOMCP_EX1", env_type)
        .def(py::init<>());

    py::class_<ccpomcp_ex2>(m, "CCPOMCP_EX2", env_type)
        .def(
            py::init<int, float, float, float, float>(),
            py::arg("length")=10,
            py::arg("small_reward")=6,
            py::arg("small_risk")=0.55,
            py::arg("large_reward")=20,
            py::arg("large_risk")=0.9
        );
}

} // end namespace rats::py
