#include "pybind/misc.hpp"
#include "pybind/agents.hpp"
#include "pybind/envs.hpp"
#include "pybind/LP_example.hpp"
#include "pybind/solvers/LP_solver.pybind.hpp"
#include "pybind/kernell.pybind.hpp"
#include "pybind/pybind.hpp"


PYBIND11_MODULE(rats, m) {
    rats::py::register_misc(m);
    rats::py::register_environments(m);
    rats::py::register_agents(m);
    rats::py::register_lp_solvers(m);
    rats::py::register_kernells(m);
    rats::py::example::register_lp_example(m);
}

