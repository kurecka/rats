#pragma once
#include "agents/agent.hpp"
#include "pybind/pybind.hpp"
#include <string>

namespace rats::py {

template <typename S, typename A>
py::class_<agent<S, A>, std::shared_ptr<agent<S, A>>> register_agent(py::module& m, std::string name) {
    py::class_<agent<S, A>, std::shared_ptr<agent<S, A>>> agent_type(m, name.c_str());
    agent_type
        .def("play", &agent<S, A>::play)
        .def("name", &agent<S, A>::name)
        .def("reset", &agent<S, A>::reset)
        .def("set_handler", py::overload_cast<environment<S, A>&>(&agent<S, A>::set_handler))
        .def("get_handler", &agent<S, A>::get_handler)
        .def("template_type", [name](const agent<S, A>& self) {
            std::string delim = "__";
            size_t pos = name.rfind(delim);
            return name.substr(pos + delim.length());
        });
    return agent_type;
}

} // end namespace rats::py
