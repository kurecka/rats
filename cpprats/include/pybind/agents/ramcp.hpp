#pragma once
#include "pybind/pybind.hpp"
#include "agents/tree_search/ramcp.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_ramcp(py::module &m, const T& agent_type, std::string name) {
    py::class_<ts::ramcp<S, A>>(m, name.c_str(), agent_type)
        .def(py::init<environment_handler<S, A>, int, float, float, float, int, int, float, bool, int>(),
        "handler"_a, "max_depth"_a, "risk_thd"_a, "gamma"_a, "gammap"_a = 1,
        "num_sim"_a = 100, "sim_time_limit"_a = 0,
        "exploration_constant"_a = 5.0,
        "rollout"_a = false,
        "graphviz_depth"_a = -1)
        .def("get_graphviz", &ts::ramcp<S, A>::get_graphviz)
        .def("get_simulations_ran", &ts::ramcp<S, A>::get_simulations_ran);
}

} // end namespace rats::py
