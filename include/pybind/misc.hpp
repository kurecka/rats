#pragma once

#include <spdlog/spdlog.h>
#include "pybind/pybind.hpp"
#include "rand.hpp"
#include "exampleConfig.h"


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


void set_rng_seed(unsigned int seed) {
    rng::set_seed(seed);
}


std::string build_info() {
    std::string info = "rats " + std::to_string(PROJECT_VERSION_MAJOR) + "." + std::to_string(PROJECT_VERSION_MINOR);
    info += " (" + std::string(__DATE__) + ", " + std::string(__TIME__) + ")";
    return info;
}


namespace rats::py {
    void register_misc(py::module& m) {
        m.def("set_log_level", &set_log_level, "Set the log level");
        m.def("set_rng_seed", &set_rng_seed, "Set the seed of the random number generator");
        m.def("build_info", &build_info, "Get the build info");
    }
}
