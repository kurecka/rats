#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/continuing_hallway.hpp"


namespace rats::py {

template <typename T>
void register_continuing_hallway(py::module &m, const T& env_type) {
    py::class_<continuing_hallway>(m, "ContHallway", env_type)
        .def(py::init<std::string, float, float>(), py::arg("map"), py::arg("trap_prob"), py::arg("slide_prob") = 0.0f)
        .def("get_width", &continuing_hallway::get_width);
}

} // end namespace rats::py
