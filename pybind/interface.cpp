#include "pybind/pybind.hpp"
#include "pybind/envs.hpp"
#include "pybind/agents.hpp"
#include "pybind/LP_example.hpp"
#include "pybind/kernell.pybind.hpp"
#include "exampleConfig.h"

// #include <fstream>
// #include "spdlog/spdlog.h"
// #include "spdlog/sinks/basic_file_sink.h"
// #include "spdlog/fmt/ostr.h"


void set_log_level(std::string level) {
    std::transform(level.begin(), level.end(), level.begin(), 
        [](unsigned char c){ return std::tolower(c); }
    );
    std::map<std::string, spdlog::level::level_enum> level_map = {
        {"trace", spdlog::level::trace},
        {"debug", spdlog::level::debug},
        {"info", spdlog::level::info},
        {"warn", spdlog::level::warn},
        {"warning", spdlog::level::warn},
        {"err", spdlog::level::err},
        {"error", spdlog::level::err},
        {"critical", spdlog::level::critical},
        {"off", spdlog::level::off}
    };
    spdlog::set_level(level_map[level]);
}

std::string build_info() {
    std::string info = "rats version: " + std::to_string(PROJECT_VERSION_MAJOR) + "." + std::to_string(PROJECT_VERSION_MINOR);
    info += "(" + std::string(__DATE__) + " " + std::string(__TIME__) + ")";
    return info;
}

// void set_graphviz_file(std::string file_name) {
//     std::ofstream file(file_name, std::ofstream::out | std::ofstream::trunc);
//     file.close();
//     auto graph_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_name);
//     auto logger = std::make_shared<spdlog::logger>("graphviz", graph_sink);
//     logger->set_pattern("%v");
//     spdlog::register_logger(logger);
// }

PYBIND11_MODULE(rats, m) {
    rats::py::register_environments(m);
    rats::py::register_agents(m);
    rats::py::register_kernells(m);

    rats::py::example::register_lp_example(m);

    m.def("set_log_level", &set_log_level, "Set the log level");
    m.def("build_info", &build_info, "Get the build info");
    // m.def("set_graphviz_file", &set_graphviz_file, "Set the graphviz file");
}
