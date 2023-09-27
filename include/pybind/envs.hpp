#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "pybind/envs/investor_env.hpp"
#include "pybind/envs/frozen_lake.hpp"
#include "pybind/envs/hallway.hpp"


namespace rats::py {

void register_environments(py::module& m) {
    auto env_type = register_environment<int, size_t>(m);
    register_investor_env(m, env_type);
    register_frozen_lake(m, env_type);
    
    auto hallway_type = register_hallway(m, env_type);
    register_hallway_agent(m, hallway_type);
}

}  // namespace rats::py
