#pragma once
#include "pybind/pybind.hpp"
#include "agents/tree_search/tuct.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_lambda_tuct(py::module &m, const T& agent_type, std::string name) {
    py::class_<ts::tuct<S, A, true>>(m, name.c_str(), agent_type)
        .def(py::init<environment_handler<S, A>, int, float, float, float, int, int, float, float, int, bool, float>(),
        "handler"_a, "max_depth"_a, "risk_thd"_a, "gamma"_a, "gammap"_a = 1,
        "num_sim"_a = 100, "sim_time_limit"_a = 0,
        "exploration_constant"_a = 5.0, "risk_exploration_ratio"_a = 1, "graphviz_depth"_a = -1,
        "use_predictor"_a = false, "lambda"_a = 50)
        .def("get_graphviz", &ts::tuct<S, A, true>::get_graphviz);
}

} // end namespace rats::py
