#pragma once
#include "pybind/pybind.hpp"
#include "agents/tree_search/ccpomcp.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_ccpomcp(py::module &m, const T& agent_type, std::string name) {
    py::class_<ts::ccpomcp<S, A>>(m, name.c_str(), agent_type)
        .def(py::init<environment_handler<S, A>, int, float, float, float, int, int, float, float, float, bool, int, int>(),
        "handler"_a, "max_depth"_a, "risk_thd"_a, "gamma"_a, "gammap"_a = 1,
        "num_sim"_a = 100, "sim_time_limit"_a = 0,
        "exploration_constant"_a = 5.0, "initial_lambda"_a = 0, "lr"_a = -1,
        "rollout"_a = false, "num_rollouts"_a = 10,
        "graphviz_depth"_a = -1)
        .def("get_graphviz", &ts::ccpomcp<S, A>::get_graphviz)
        .def("get_simulations_ran", &ts::ccpomcp<S, A>::get_simulations_ran);
}

} // end namespace rats::py
