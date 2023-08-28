#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "pybind/envs/investor_env.hpp"


namespace rats::py {

void register_environments(py::module& m) {
    register_environment<int, size_t>(m);

    register_investor_env(m);
}

}  // namespace rats::py
