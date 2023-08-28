#pragma once

#include "pybind/pybind.hpp"
#include "envs/investor_env.hpp"

namespace rats::py {

void register_investor_env(py::module &m) {
    py::class_<investor_env, environment<int, size_t>>(m, "InvestorEnv")
        .def(py::init<int, int>());
}

} // end namespace rats::py
