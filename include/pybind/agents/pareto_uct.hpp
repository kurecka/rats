#pragma once
#include "pybind/pybind.hpp"
#include "agents/tree_search/pareto_uct.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_pareto_uct(py::module &m, const T& agent_type) {
    py::class_<ts::pareto_uct<S, A>>(m, "ParetoUCT", agent_type)
        .def(py::init<environment_handler<S, A>, int, int, float, float, float, int>(),
        "handler"_a, "max_depth"_a, "num_sim"_a, "risk_thd"_a, "gamma"_a,
        "exploration_constant"_a = 5.0, "graphviz_depth"_a = -1)
        .def("get_graphviz", &ts::pareto_uct<S, A>::get_graphviz);
}

} // end namespace rats::py
