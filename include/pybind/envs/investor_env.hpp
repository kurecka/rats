#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "envs/investor_env.hpp"


namespace rats::py {

template <typename T>
void register_investor_env(py::module &m, const T& env_type) {
    py::class_<investor_env>(m, "InvestorEnv", env_type)
        .def(py::init<int, int>());
}

} // end namespace rats::py
