#pragma once

#include "tree_search.hpp"
#include "../../string_utils.hpp"
#include <fstream>
#include <vector>

namespace rats { 
    template<typename ...Args>
    std::string to_string(const ts::state_node<Args...>& node) {
        std::string str = "State ("+to_string(node.state)+"):\n";
        if (node.is_leaf()) {
            str += "leaf\\n";
        }
        if (node.is_leaf_state()) {
            str += "leaf state\\n";
        }
        str += "N=" + to_string(node.num_visits);
        str += "\\nr=" + to_string(node.observed_reward);
        str += ", p=" + to_string(node.observed_penalty);
        
        // TODO: fix this to_srinGG (it does not compile with to_string)
        str += "\\nE[V]=" + to_string(node.v);
        return str;
    }

    template<typename ...Args>
    std::string to_string(const ts::action_node<Args...>& node) {
        std::string str = "Action (" + to_string(node.action) + "):\\n";
        str += "N=" + to_string(node.num_visits);
        str += "\\nE[V]=" + to_string(node.q);
        return str;
    }

    namespace ts {

    template<typename ...Args>
    std::string to_graphviz(const state_node<Args...>& node, std::string parent, size_t& id, int depth) {
        if (depth == 0) return "";
        std::string name = "state_" + to_string(id++);

        std::string str = name + " [label=\"" + to_string(node) + "\"]\n";
        if (parent.size()) {
            str += parent + " -> " + name + "\n";
        }
        
        for (const auto& child : node.children) {
            str += to_graphviz(child, name, id, depth-1);
        }

        return str;
    }

    template<typename ...Args>
    std::string to_graphviz(const action_node<Args...>& node, std::string parent, size_t& id, int depth) {
        if (depth == 0) return "";
        std::string name = "action_" + to_string(id++);

        std::string str = name + " [label=\"" + to_string(node) + "\"]\n";
        if (parent.size()) {
            str += parent + " -> " + name + "\n";
        }
        
        for (const auto& child : node.children) {
            str += to_graphviz(*child.second, name, id, depth-1);
        }

        return str;
    }

    template<typename ...Args>
    std::string to_graphviz_tree(const state_node<Args...>& node, int depth = -1) {
        std::string str = "digraph G {\n";
        size_t id = 0;
        str += to_graphviz(node, "", id, depth) + "}\n";
        return str;
    }

    } // namespace ts
} // namespace rats
