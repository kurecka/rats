#include "envs/env.hpp"
#include "envs/investor_env.hpp"

#ifdef PYBIND
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif


namespace rats {
#ifdef PYBIND
template <typename S, typename A>
void register_environment(py::module& m) {
    py::class_<environment<S, A>>(m, "Environment")
        .def("name", &environment<S, A>::name)
        .def("num_actions", &environment<S, A>::num_actions)
        .def("possible_actions", &environment<S, A>::possible_actions)
        .def("play_action", &environment<S, A>::play_action)
        .def("is_over", &environment<S, A>::is_over)
        .def("current_state", &environment<S, A>::current_state)
        .def("restore_checkpoint", &environment<S, A>::restore_checkpoint)
        .def("make_checkpoint", &environment<S, A>::make_checkpoint)
        .def("reset", &environment<S, A>::reset);
    
    py::class_<environment_handler<S, A>>(m, "EnvironmentHandler")
        .def(py::init<environment<S, A>&>())
        .def("get_reward", &environment_handler<S, A>::get_reward)
        .def("get_penalty", &environment_handler<S, A>::get_penalty)
        .def("get_num_steps", &environment_handler<S, A>::get_num_steps)
        .def("reset", &environment_handler<S, A>::reset)
        .def("play_action", &environment_handler<S, A>::play_action)
        .def("sim_action", &environment_handler<S, A>::sim_action)
        .def("sim_reset", &environment_handler<S, A>::sim_reset)
        .def("reward_range", &environment_handler<S, A>::reward_range)
        .def("num_actions", &environment_handler<S, A>::num_actions)
        .def("possible_actions", &environment_handler<S, A>::possible_actions)
        .def("get_action", &environment_handler<S, A>::get_action)
        .def("get_current_state", &environment_handler<S, A>::get_current_state);
}

void register_environments(py::module& m) {
    register_environment<int, size_t>(m);

    register_investor_env(m);
}
#endif
}  // namespace rats
