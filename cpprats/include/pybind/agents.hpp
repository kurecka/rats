#pragma once
#include "pybind/pybind.hpp"
#include "pybind/agents/agent.hpp"
#include "pybind/agents/constant_agent.hpp"
#include "pybind/agents/randomized_agent.hpp"
#include "pybind/agents/tuct.hpp"
#include "pybind/agents/lambda_tuct.hpp"
#include "pybind/agents/ccpomcp.hpp"
#include "pybind/agents/dual_ramcp.hpp"
#include "pybind/agents/primal_uct.hpp"
#include "pybind/agents/ramcp.hpp"


namespace rats::py {

template <typename S, typename A>
void register_agents_t(py::module& m, std::string type) {
    auto agent_type = register_agent<S, A>(m, "Agent" + type);
    register_constant_agent<S, A>(m, agent_type, "ConstantAgent" + type);
    register_randomized_agent<S, A>(m, agent_type, "RandomizedAgent" + type);
    register_ccpomcp<S, A>(m, agent_type, "CCPOMCP" + type);
    register_dual_ramcp<S, A>(m, agent_type, "DualRAMCP" + type);
    register_primal_uct<S, A>(m, agent_type, "PrimalUCT" + type);
    register_tuct<S, A>(m, agent_type, "TUCT" + type);
    register_lambda_tuct<S, A>(m, agent_type, "LambdaTUCT" + type);
    register_ramcp<S, A>(m, agent_type, "RAMCP" + type);
}

void register_agents(py::module& m) {
    register_agents_t<int, size_t>(m, "__<int, size_t>");
    register_agents_t<std::pair<int,uint64_t>, size_t>(m, "__<<int,uint64_t>, size_t>");
    register_agents_t<std::tuple<std::string, std::map< std::string, float >, bool>, int>(m, "__<std::tuple<std::string, std::map<std::string, float>, bool>, int>");
}
}  // namespace rats
