#pragma once
#include "pybind/pybind.hpp"
#include "pybind/agents/agent.hpp"
#include "pybind/agents/constant_agent.hpp"
#include "pybind/agents/randomized_agent.hpp"
#include "pybind/agents/dual_uct.hpp"
#include "pybind/agents/primal_uct.hpp"
#include "pybind/agents/pareto_uct.hpp"


namespace rats::py {

template <typename S, typename A>
void register_agents_t(py::module& m, std::string type) {
    auto agent_type = register_agent<S, A>(m, "Agent" + type);
    register_constant_agent<S, A>(m, agent_type, "ConstantAgent" + type);
    register_randomized_agent<S, A>(m, agent_type, "RandomizedAgent" + type);
    register_dual_uct<S, A>(m, agent_type, "DualUCT" + type);
    register_primal_uct<S, A>(m, agent_type, "PrimalUCT" + type);
    register_pareto_uct<S, A>(m, agent_type, "ParetoUCT" + type);
}

void register_agents(py::module& m) {
    register_agents_t<int, size_t>(m, "__<int, size_t>");
    register_agents_t<std::pair<int,uint64_t>, size_t>(m, "__<<int,uint64_t>, size_t>");
}

}  // namespace rats
