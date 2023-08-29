#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "pybind/envs/investor_env.hpp"
#include "pybind/envs/frozen_lake.hpp"


namespace rats::py {

void register_environments(py::module& m) {
    auto env_type = register_environment<int, size_t>(m);

    register_investor_env(m, env_type);
    register_frozen_lake(m, env_type);
}

}  // namespace rats::py
