#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/manhattan.hpp"
#include <string>


namespace rats::py {

template <typename T>
void register_manhattan(py::module &m, const T& env_type) {
    py::class_<manhattan>(m, "Manhattan", env_type)
        .def(py::init<float, std::vector< std::string >, std::vector< std::string >, std::string >(), py::arg("capacity"), py::arg("targets"), py::arg("reloads"), py::arg("init_state"));
}

} // end namespace rats::py
