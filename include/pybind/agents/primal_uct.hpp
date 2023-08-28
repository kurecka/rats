#pragma once
#include "pybind/pybind.hpp"
#include "agents/tree_search/primal_uct.hpp"

namespace rats::py {

template <typename S, typename A>
void register_primal_uct(py::module &m) {
    py::class_<ts::primal_uct<S, A>, agent<S, A>>(m, "PrimalUCT")
        .def(py::init<environment_handler<S, A>, int, int, float, float, float>(),
        "handler"_a, "max_depth"_a, "num_sim"_a, "risk_thd"_a, "gamma"_a, "exploration_constant"_a = 5.0);
}

} // end namespace rats::py
