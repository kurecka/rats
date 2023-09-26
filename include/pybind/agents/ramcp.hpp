#pragma once
#include "pybind/pybind.hpp"
#include "agents/tree_search/ramcp.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_ramcp(py::module &m, const T& agent_type) {
    py::class_<ts::ramcp<S, A>>(m, "RAMCP", agent_type)
        .def(py::init<environment_handler<S, A>, int, int, float, float, float>(),
        "handler"_a, "max_depth"_a, "num_sim"_a, "risk_thd"_a, "gamma"_a, "exploration_constant"_a = 5.0);
}

} // end namespace rats::py
