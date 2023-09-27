#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/hallway.hpp"


namespace rats::py {

template <typename T>
void register_hallwat(py::module &m, const T& env_type) {
    py::class_<frozen_lake>(m, "Hallway", env_type)
        .def(py::init<std::string, float>());
}

} // end namespace rats::py
