// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef ENABLE_DOCTEST_IN_LIBRARY
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include <iostream>
#include <stdlib.h>
#include <map>

#include "rand.hpp"
#include "exampleConfig.h"
#include "kernell.hpp"
#include "envs.hpp"
#include "agents.hpp"
#include "spdlog/spdlog.h"

struct arg_spec {
    std::string name;
    std::string verbose_name;
    std::string description;
    std::string default_value;
    bool is_flag;
};

std::vector<arg_spec> get_arg_spec() {
    std::vector<arg_spec> spec;
    spec.push_back({"-d", "--depth", "max depth of the MC tree", "10", 0});
    spec.push_back({"-n", "--num_sim", "number of simulation runs per decision", "100", 0});
    spec.push_back({"-e", "--num_episodes", "number of episodes", "100", 0});
    spec.push_back({"-r", "--risk_thd", "risk threshold", "0.1", 0});
    spec.push_back({"-s", "--seed", "seed of the random number generator", "-1", 0});
    spec.push_back({"-l", "--loglevel", "log level", "INFO", 0});
    spec.push_back({"-x", "--expl_const", "exploration constant", "1", 0});

    spec.push_back({"-v", "--version", "print version", "false", 1});
    spec.push_back({"-h", "--help", "print this help", "false", 1});
    return spec;
}

std::string get_help(const std::vector<arg_spec>& specs) {
    std::string help = "Usage: ralph [options]\n";
    help += "Options:\n";

    size_t verb_name_len = 0;
    for (auto spec : specs) verb_name_len = std::max(verb_name_len, spec.verbose_name.length());

    for (auto spec : specs) {
        help += spec.name + "\t" + spec.verbose_name;
        for (size_t i = 0; i < verb_name_len - spec.verbose_name.length(); i++) help += " ";
        help += "\t" + spec.description;
        if (!spec.is_flag) help += " (default: " + spec.default_value + ")";
        help += "\n";
    }
    return help;
}

std::string get_version() {
    std::string version = "Ralph version " + std::to_string(PROJECT_VERSION_MAJOR);
    return version;
}

/*
  A function loading input arguments from the command line in format -arg_name arg_value
*/
std::map<std::string, std::string> load_args(int argc, char *argv[], const std::vector<arg_spec>& specs) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        // Identify the argument
        std::string parsed_name(argv[i]);
        std::string arg_name;
        arg_spec spec;
        for (auto& s : specs) {
            if (s.verbose_name == parsed_name || s.name == parsed_name) {
                spec = s;
                arg_name = spec.verbose_name;
                break;
            }
        }
        if (arg_name.empty()) {
            std::cout << "Unknown argument: " << parsed_name << std::endl;
            std::cout << get_help(specs);
            exit(1);
        }

        // Check if the argument has the correct number of values
        if (!spec.is_flag && !(argc-i-1)) {
            std::cout << "Argument " << parsed_name << " requires a value" << std::endl;
            std::cout << get_help(specs);
            exit(1);
        }

        if (spec.is_flag) {
            args[arg_name] = "true";
        } else {
            args[arg_name] = argv[++i];
        }
    }

    // Set default values
    for (auto arg : specs) {
        if (args.find(arg.verbose_name) == args.end()) {
            args[arg.verbose_name] = arg.default_value;
        }
    }
    return args;
}

int main(int argc, char *argv[]) {
    rng::init();

    std::vector arg_spec = get_arg_spec();
    std::map<std::string, std::string> args = load_args(argc, argv, arg_spec);
    
    if (args["--help"] == "true") {
        std::cout << get_help(arg_spec) << std::endl;
        return 0;
    }

    if (args["--version"] == "true") {
        std::cout << get_version() << std::endl;
        return 0;
    }

    if (args["--loglevel"] != "") {
        std::string level = args["--loglevel"];
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

    spdlog::info("Arguments:");
    for (auto arg : args) {
        if (arg.first == "--help" || arg.first == "--verbose" || arg.first == "--version") continue;
        spdlog::info(arg.first + " = " + arg.second);
    }

    if (std::stoi(args["--seed"]) >= 0) {
        // string to unsigned int
        // set_seed(static_cast<size_t>(std::stoi(args["--seed"])));
    }

    int num_episodes = std::stoi(args["--num_episodes"]);

    // Run the simulation
    int initial_state = 2;

    orchestrator<int, size_t> o;
    o.load_environment(new gym::investor_env(initial_state, 20));

    o.load_agent(new gym::randomized_agent<int, size_t>());
    o.run(num_episodes, 0);

    o.load_agent(new gym::constant_agent<int, size_t>(0));
    o.run(num_episodes, 0);

    o.load_agent(new gym::constant_agent<int, size_t>(1));
    o.run(num_episodes, 0);

    o.load_agent(new gym::ts::primal_uct<int, size_t>(
        std::stoi(
            args["--depth"]),
            std::stoi(args["--num_sim"]),
            std::stof(args["--risk_thd"]),
            0.9f,
            std::stof(args["--expl_const"])
        )
    );
    o.run(num_episodes, 0);

    o.load_agent(new gym::ts::dual_uct<int, size_t>(
        std::stoi(args["--depth"]),
        std::stoi(args["--num_sim"]),
        std::stof(args["--risk_thd"]),
        0.9f,
        std::stof(args["--expl_const"])
    ));
    o.run(num_episodes, 0);
}
