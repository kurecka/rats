#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/ccpomcp_ex1.hpp"


namespace rats::py {

template <typename T>
void register_ccpomcp_ex1(py::module &m, const T& env_type) {
    py::class_<ccpomcp_ex1>(m, "CCPOMCP_EX1", env_type)
        .def(py::init<>());
}

} // end namespace rats::py
