#include "agents/constant_agent.hpp"
#include "agents/randomized_agent.hpp"
#include "agents/tree_search/primal_uct.hpp"
#include "agents/tree_search/dual_uct.hpp"
// #include "agents/tree_search/pareto_uct.hpp"

#ifdef PYBIND
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif


namespace rats {
#ifdef PYBIND
template <typename S, typename A>
void register_agent(py::module& m) {
    py::class_<agent<S, A>>(m, "Agent")
        .def("play", &agent<S, A>::play)
        .def("set_handler", py::overload_cast<environment<S, A>&>(&agent<S, A>::set_handler));
}

void register_agents(py::module& m) {
    register_agent<int, size_t>(m);

    register_constant_agent<int, size_t>(m);
}
#endif
}  // namespace rats
