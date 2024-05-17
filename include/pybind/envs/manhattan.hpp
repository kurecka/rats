#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/manhattan.hpp"
#include <string>


namespace rats::py {

template <typename T>
void register_manhattan(py::module &m, const T& env_type) {
    py::class_<manhattan>(m, "Manhattan", env_type)
        .def(py::init<float, std::vector< std::string >, std::map< std::string, float >, std::string, float >(),
         py::arg("capacity"), py::arg("targets"), py::arg("periods"), py::arg("init_state") = "", py::arg("cons_thd") = 10.0f)
        .def("animate_simulation", &manhattan::animate_simulation, py::arg("interval") = 100, py::arg("filename") = "map.html");
}

} // end namespace rats::py
