#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/frozen_lake.hpp"


namespace rats::py {

template <typename T>
void register_frozen_lake(py::module &m, const T& env_type) {
    py::class_<frozen_lake>(m, "FrozenLake", env_type)
        .def(py::init<>());
}

} // end namespace rats::py
