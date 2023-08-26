#define PYBIND

#include <pybind11/pybind11.h>
#include "envs.hpp"
#include "agents.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rats, m) {
    rats::register_environments(m);
    rats::register_agents(m);
}
