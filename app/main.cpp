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
#include "world.hpp"

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
    spec.push_back({"-r", "--risk_thd", "risk threshold", "0.1", 0});
    spec.push_back({"-s", "--seed", "seed of the random number generator", "-1", 0});

    spec.push_back({"-v", "--version", "print version", "false", 1});
    spec.push_back({"", "--verbose", "enable verbose mode", "false", 1});
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
    std::vector arg_spec = get_arg_spec();
    std::map<std::string, std::string> args = load_args(argc, argv, arg_spec);
    
    if (args["--help"] == "true") {
        std::cout << get_help(arg_spec);
        return 0;
    }

    if (args["--version"] == "true") {
        std::cout << get_version();
        return 0;
    }

    if (args["--verbose"] == "true") {
        std::cout << "Verbose mode enabled" << std::endl;
        std::cout << "Arguments:" << std::endl;
        for (auto arg : args) {
            if (arg.first == "--help" || arg.first == "--verbose" || arg.first == "--version") continue;
            std::cout << arg.first << " = " << arg.second << std::endl;
        }
    }

    if (std::stoi(args["--seed"]) >= 0) {
        set_seed(std::stoi(args["--seed"]));
    }

    // Run the simulation
    int initial_state = 5;
    world::investor_env game(initial_state, 20);
    world::random_agent<int, int> a(game.get_state(), game.get_action_space());

    while (!game.is_over()) {
        a.pass_outcome(game.play_action(a.get_action()));
        std::cout << "State: " << game.get_state() << std::endl;
    }
}
