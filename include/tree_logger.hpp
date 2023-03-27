#pragma once

#include <iostream>
#include <vector>

class tree_logger {
private:
    std::vector<std::string> logs;
    int id = 0;

    template <typename T>
    std::string log_state(const T& root, std::string parent_name) {
        std::string name = "state_" + std::to_string(id++);

        std::string log = name + " [label=\"" + root.to_string() + "\"]\n";
        if (parent_name.size()) {
            log += parent_name + " -> " + name + "\n";
        }
        
        for (const auto& child : root.children) {
            log += log_action(child, name);
        }

        return log;
    }

    template <typename T>
    std::string log_action(const T& root, std::string parent_name) {
        std::string name = "state_" + std::to_string(id++);

        std::string log = name + " [label=\"" + root.to_string() + "\"]\n";
        if (parent_name.size()) {
            log += parent_name + " -> " + name + "\n";
        }
        
        for (const auto& [key, child] : root.children) {
            log += log_state(child, name);
        }

        return log;
    }

public:
    template <typename T>
    void log(const T& root) {
        logs.push_back(log_state(root, ""));
    }

    std::string dump() const {
        std::string log = "digraph G {\n";
        for (const auto& l : logs) {
            log += l;
        }
        log += "}\n";
        return log;
    }

    void clear() {
        logs.clear();
        id = 0;
    }
};