#pragma once
#include "pybind/pybind.hpp"
#include "agents/constant_agent.hpp"

namespace rats::py {

template <typename S, typename A>
void register_constant_agent(py::module &m) {
    py::class_<constant_agent<S, A>, agent<S, A>>(m, "ConstantAgent")
        .def(py::init<environment_handler<S, A>, A>(), "handler"_a, "action"_a);
}

} // end namespace rats::py
