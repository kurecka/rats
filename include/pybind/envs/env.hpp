#pragma once

#include "pybind/pybind.hpp"
#include "envs/env.hpp"

namespace rats::py {

template <typename S, typename A>
py::class_<environment<S, A>, std::shared_ptr<environment<S, A>>> register_environment(py::module& m, std::string type_name) {
    py::class_<environment<S, A>, std::shared_ptr<environment<S, A>>> env_type(m, ("Environment" + type_name).c_str());
    env_type
        .def("name", &environment<S, A>::name)
        .def("num_actions", &environment<S, A>::num_actions)
        .def("possible_actions", &environment<S, A>::possible_actions)
        .def("play_action", &environment<S, A>::play_action)
        .def("is_over", &environment<S, A>::is_over)
        .def("current_state", &environment<S, A>::current_state)
        .def("restore_checkpoint", &environment<S, A>::restore_checkpoint)
        .def("make_checkpoint", &environment<S, A>::make_checkpoint)
        .def("reset", &environment<S, A>::reset)
        .def("template_type", [type_name](const environment<S, A>& self) {
            std::string delim = "__";
            size_t pos = type_name.rfind(delim);
            return type_name.substr(pos + delim.length());
        });
    
    py::class_<environment_handler<S, A>>(m, ("EnvironmentHandler" + type_name).c_str())
        .def(py::init<environment<S, A>&>())
        .def("get_reward", &environment_handler<S, A>::get_reward)
        .def("get_penalty", &environment_handler<S, A>::get_penalty)
        .def("get_num_steps", &environment_handler<S, A>::get_num_steps)
        .def("reset", &environment_handler<S, A>::reset)
        .def("play_action", &environment_handler<S, A>::play_action)
        .def("sim_action", &environment_handler<S, A>::sim_action)
        .def("end_sim", &environment_handler<S, A>::end_sim)
        .def("reward_range", &environment_handler<S, A>::reward_range)
        .def("num_actions", &environment_handler<S, A>::num_actions)
        .def("possible_actions", &environment_handler<S, A>::possible_actions)
        .def("get_action", &environment_handler<S, A>::get_action)
        .def("get_current_state", &environment_handler<S, A>::get_current_state)
        .def("is_over", &environment_handler<S, A>::is_over);

    return env_type;
}

} // end namespace rats::py
