#pragma once
#include "pybind/pybind.hpp"
#include "agents/randomized_agent.hpp"

namespace rats::py {

template <typename S, typename A>
void register_randomized_agent(py::module &m) {
    py::class_<randomized_agent<S, A>, agent<S, A>>(m, "RandomizedAgent")
        .def(py::init<environment_handler<S, A>>());
}

} // end namespace rats::py
