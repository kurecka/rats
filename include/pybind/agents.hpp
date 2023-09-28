#pragma once
#include "pybind/pybind.hpp"
#include "pybind/agents/agent.hpp"
#include "pybind/agents/constant_agent.hpp"
#include "pybind/agents/randomized_agent.hpp"
// #include "pybind/agents/dual_uct.hpp"
// #include "pybind/agents/primal_uct.hpp"
#include "pybind/agents/pareto_uct.hpp"


namespace rats::py {

template <typename S, typename A>
void register_agents_t(py::module& m) {
    auto agent_type = register_agent<S, A>(m);
    register_constant_agent<S, A>(m, agent_type);
    register_randomized_agent<S, A>(m, agent_type);
    // register_dual_uct<S, A>(m, agent_type);
    // register_primal_uct<S, A>(m, agent_type);
    // register_pareto_uct<S, A>(m, agent_type);
}

void register_agents(py::module& m) {
    register_agents_t<int, size_t>(m);
    register_agents_t<std::pair<int,uint64_t>, size_t>(m);
}

}  // namespace rats
