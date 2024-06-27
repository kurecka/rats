#pragma once
#include "pybind/pybind.hpp"
#include "agents/randomized_agent.hpp"

namespace rats::py {

template <typename S, typename A, typename T>
void register_randomized_agent(py::module &m, const T &agent_type, std::string name) {
    py::class_<randomized_agent<S, A>>(m, name.c_str(), agent_type)
        .def(py::init<environment_handler<S, A>>());
}

} // end namespace rats::py
