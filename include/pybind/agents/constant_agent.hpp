#pragma once
#include "pybind/pybind.hpp"
#include "agents/constant_agent.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_constant_agent(py::module &m, const T& agent_type) {
    py::class_<constant_agent<S, A>>(m, "ConstantAgent", agent_type)
        .def(py::init<environment_handler<S, A>, A>(), "handler"_a, "action"_a);
}

} // end namespace rats::py
