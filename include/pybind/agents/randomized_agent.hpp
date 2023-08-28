#pragma once
#include "pybind/pybind.hpp"
#include "agents/randomized_agent.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_randomized_agent(py::module &m, const T &agent_type) {
    py::class_<randomized_agent<S, A>>(m, "RandomizedAgent", agent_type)
        .def(py::init<environment_handler<S, A>>());
}

} // end namespace rats::py
