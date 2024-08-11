#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/soft_avoid.hpp"


namespace rats::py {

template <typename T>
void register_soft_avoid(py::module &m, const T& env_type) {
    py::class_<soft_avoid>(m, "SoftAvoid", env_type)
        .def(py::init<std::string, float, float>(), py::arg("map"), py::arg("trap_prob"), py::arg("slide_prob") = 0.0f)
        .def("get_width", &soft_avoid::get_width);
}

} // end namespace rats::py
