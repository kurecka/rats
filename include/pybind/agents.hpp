#pragma once
#include "pybind/pybind.hpp"
#include "pybind/agents/agent.hpp"
#include "pybind/agents/constant_agent.hpp"
#include "pybind/agents/randomized_agent.hpp"
#include "pybind/agents/dual_uct.hpp"
#include "pybind/agents/primal_uct.hpp"

namespace rats::py {

void register_agents(py::module& m) {
    register_agent<int, size_t>(m);
    register_constant_agent<int, size_t>(m);
    register_randomized_agent<int, size_t>(m);
    register_dual_uct<int, size_t>(m);
    register_primal_uct<int, size_t>(m);
}

}  // namespace rats
