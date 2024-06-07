#pragma once
#include "pybind/pybind.hpp"
#include "agents/tree_search/primal_uct.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_primal_uct(py::module &m, const T& agent_type, std::string name) {
    py::class_<ts::primal_uct<S, A>>(m, name.c_str(), agent_type)
        .def(py::init<environment_handler<S, A>, int, float, float, float, int, int, float>(),
        "handler"_a, "max_depth"_a, "risk_thd"_a, "gamma"_a, "gammap"_a = 1,
        "num_sim"_a = 100, "sim_time_limit"_a = 0,
        "exploration_constant"_a = 5.0);
}

} // end namespace rats::py
