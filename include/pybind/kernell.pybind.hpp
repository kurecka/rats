#pragma once

#include "pybind/pybind.hpp"
#include "kernell.hpp"

namespace rats::py {
    template <typename S, typename A>
    void register_kernell(py::module& m) {
        py::class_<orchestrator<S, A>>(m, "Orchestrator")
            .def(py::init<>())
            .def("load_environment",
            [](orchestrator<S, A> &o, std::shared_ptr<environment<S, A>> e) {
                o.load_environment(e);
            })
            .def("load_agent",
            [](orchestrator<S, A> &o, std::shared_ptr<agent<S, A>> a) {
                o.load_agent(a);
            })
            .def(
                "run", &orchestrator<S, A>::run,
                "num_episodes"_a, "num_train_episodes"_a = 0
            );
    }

    void register_kernells(py::module& m) {
        register_kernell<int, size_t>(m);
    }
}
