#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "pybind/envs/investor_env.hpp"
#include "pybind/envs/frozen_lake.hpp"
#include "pybind/envs/manhattan.hpp"
#include "pybind/envs/hallway.hpp"
#include "pybind/envs/ccpomcp_ex.hpp"


namespace rats::py {

void register_environments(py::module& m) {
    auto env_type = register_environment<int, size_t>(m, "__<int, size_t>");

    auto manhattan_type = register_environment<std::string, size_t>(m, "__<std::string, size_t>");

    register_investor_env(m, env_type);
    register_manhattan(m, manhattan_type);
    register_ccpomcp_ex(m, env_type);
    register_frozen_lake(m, env_type);
    
    auto hallway_type = register_environment<std::pair<int,uint64_t>, size_t>(m, "__<<int,uint64_t>, size_t>");
    register_hallway(m, hallway_type);
}

}  // namespace rats::py
