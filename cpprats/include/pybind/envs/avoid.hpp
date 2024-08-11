#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/avoid.hpp"


namespace rats::py {

template <typename T>
void register_avoid(py::module &m, const T& env_type) {
    py::class_<avoid>(m, "Avoid", env_type)
        .def(py::init<std::string, float, float>(), py::arg("map"), py::arg("trap_prob"), py::arg("slide_prob") = 0.0f)
        .def("get_width", &avoid::get_width);
}

} // end namespace rats::py
