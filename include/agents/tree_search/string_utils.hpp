#pragma once

#include "tree_search.hpp"
#include <iostream>
#include <vector>

namespace rats::ts {
    std::string to_string(const std::pair<float, float>& p) {
        return "(" + std::to_string(p.first) + "," + std::to_string(p.second) + ")";
    }

    std::string to_string(int i) {
        return std::to_string(i);
    }

    std::string to_string(float f) {
        return std::to_string(f);
    }

    std::string to_string(double d) {
        return std::to_string(d);
    }

    std::string to_string(size_t s) {
        return std::to_string(s);
    }

    template<typename S, typename A, typename DATA, typename V, typename Q>
    std::string to_string(const state_node<S, A, DATA, V, Q>& node) {
        std::string str = "State ("+to_string(node.state)+"):\n";
        str += "N=" + std::to_string(node.num_visits);
        str += "\\nr=" + std::to_string(node.observed_reward);
        str += ", p=" + std::to_string(node.observed_penalty);
        
        str += "\\nE[V]=" + to_string(node.v);
        return str;
    }

    template<typename S, typename A, typename DATA, typename V, typename Q>
    std::string to_string(const action_node<S, A, DATA, V, Q>& node) {
        std::string str = "Action (" + to_string(node.action) + "):\\n";
        str += "N=" + std::to_string(node.num_visits);
        str += "\\nE[V]=" + to_string(node.q);
        return str;
    }

    template<typename S, typename A, typename DATA, typename V, typename Q>
    std::string to_graphviz(const state_node<S, A, DATA, V, Q>& node, std::string parent, size_t& id, int depth) {
        if (depth == 0) return "";
        std::string name = "state_" + std::to_string(id++);

        std::string str = name + " [label=\"" + to_string(node) + "\"]\n";
        if (parent.size()) {
            str += parent + " -> " + name + "\n";
        }
        
        for (const auto& child : node.children) {
            str += to_graphviz(child, name, id, depth-1);
        }

        return str;
    }

    template<typename S, typename A, typename DATA, typename V, typename Q>
    std::string to_graphviz(const action_node<S, A, DATA, V, Q>& node, std::string parent, size_t& id, int depth) {
        if (depth == 0) return "";
        std::string name = "action_" + std::to_string(id++);

        std::string str = name + " [label=\"" + to_string(node) + "\"]\n";
        if (parent.size()) {
            str += parent + " -> " + name + "\n";
        }
        
        for (const auto& child : node.children) {
            str += to_graphviz(*child.second, name, id, depth-1);
        }

        return str;
    }

    template<typename S, typename A, typename DATA, typename V, typename Q>
    std::string to_graphviz_tree(const state_node<S, A, DATA, V, Q>& node, int depth = -1) {
        std::string str = "digraph G {\n";
        size_t id = 0;
        str += to_graphviz(node, "", id, depth) + "}\n";
        return str;
    }
};
