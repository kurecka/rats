#include "pybind/pybind.hpp"
#include "pybind/envs.hpp"
#include "pybind/agents.hpp"
#include "pybind/LP_example.hpp"
#include "pybind/kernell.pybind.hpp"


void set_log_level(std::string level) {
    std::transform(level.begin(), level.end(), level.begin(), 
        [](unsigned char c){ return std::tolower(c); }
    );
    std::map<std::string, spdlog::level::level_enum> level_map = {
        {"trace", spdlog::level::trace},
        {"debug", spdlog::level::debug},
        {"info", spdlog::level::info},
        {"warn", spdlog::level::warn},
        {"err", spdlog::level::err},
        {"critical", spdlog::level::critical},
        {"off", spdlog::level::off}
    };
    spdlog::set_level(level_map[level]);
}

PYBIND11_MODULE(rats, m) {
    rats::py::register_environments(m);
    rats::py::register_agents(m);
    rats::py::register_kernells(m);

    rats::py::example::register_lp_example(m);

    m.def("set_log_level", &set_log_level, "Set the log level");
}
