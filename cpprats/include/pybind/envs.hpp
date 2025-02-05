#pragma once

#include "pybind/pybind.hpp"
#include "pybind/envs/env.hpp"
#include "pybind/envs/investor_env.hpp"
#include "pybind/envs/frozen_lake.hpp"
#include "pybind/envs/manhattan.hpp"
#include "pybind/envs/avoid.hpp"
#include "pybind/envs/soft_avoid.hpp"
#include "pybind/envs/ccpomcp_ex.hpp"


namespace rats::py {

void register_environments(py::module& m) {
    auto env_type = register_environment<int, size_t>(m, "__<int, size_t>");

    auto manhattan_type = register_environment< std::tuple<std::string, std::map< std::string, float >, bool>, int>(m, "__<std::tuple<std::string, std::map<std::string, float>, bool>, int>");

    register_investor_env(m, env_type);
    register_manhattan(m, manhattan_type);
    register_ccpomcp_ex(m, env_type);
    register_frozen_lake(m, env_type);
    
    auto avoid_type = register_environment<std::pair<int,uint64_t>, size_t>(m, "__<<int,uint64_t>, size_t>");
    register_avoid(m, avoid_type);
    register_soft_avoid(m, avoid_type);
}

}  // namespace rats::py
