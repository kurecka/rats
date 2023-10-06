#pragma once
#include "pybind/pybind.hpp"
#include "agents/tree_search/dual_uct.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_dual_uct(py::module &m, const T& agent_type, std::string name) {
    py::class_<ts::dual_uct<S, A>>(m, name.c_str(), agent_type)
        .def(py::init<environment_handler<S, A>, int, float, float, float, int, int, float, float, float, int>(),
        "handler"_a, "max_depth"_a, "risk_thd"_a, "gamma"_a, "gammap"_a = 1,
        "num_sim"_a = 100, "sim_time_limit"_a = 0,
        "exploration_constant"_a = 5.0, "initial_lambda"_a = 0, "lr"_a = 0.0005,
        "graphviz_depth"_a = -1)
        .def("get_graphviz", &ts::dual_uct<S, A>::get_graphviz);
}

} // end namespace rats::py