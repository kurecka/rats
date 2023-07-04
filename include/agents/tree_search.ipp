#pragma once

#include <vector>
#include <map>
#include <cassert>


namespace gym {
namespace ts {

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
SN* tree_search<S, A, SN, AN>::select() {
    spdlog::trace("Selecting node");
    state_node_t* current_state_node = root.get();

    int depth = 0;

    float run_risk_thd = step_risk_thd;

    while (!current_state_node->is_leaf() && depth < max_depth && !current_state_node->is_terminal) {
        A action = current_state_node->select_action(run_risk_thd, true);
        action_node_t *current_action_node = &current_state_node->children[action];
        auto [s, r, p, t] = agent<S, A>::handler.sim_action(action);
        current_action_node->add_outcome(s, r, p, t);
        current_state_node = current_action_node->children[s].get();

        int state_visits = current_state_node->num_visits;
        int action_visits = current_action_node->num_visits;

        run_risk_thd *= action_visits / static_cast<float>(state_visits + 0.0001);

        depth++;
    }

    agent<S, A>::handler.sim_reset();
    return current_state_node;
}

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
void tree_search<S, A, SN, AN>::expand(SN* leaf) {
    spdlog::trace("Expanding node");
    leaf->expand(agent<S, A>::handler.num_actions(agent<S, A>::handler.current_state()));
}

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
void tree_search<S, A, SN, AN>::propagate(state_node_t* leaf) {
    spdlog::trace("Propagatin gresults");
    action_node_t* prev_action_node = nullptr;
    state_node_t* current_state_node = leaf;

    while (!current_state_node->is_root()) {
        current_state_node->propagate(prev_action_node, gamma);
        action_node_t* current_action_node = current_state_node->parent;
        current_action_node->propagate(current_state_node, gamma);
        current_state_node = current_action_node->parent;
        prev_action_node = current_action_node;
    }

    current_state_node->propagate(prev_action_node, gamma);
}

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
void tree_search<S, A, SN, AN>::descent(A a, S s) {
    spdlog::trace("Descending tree");
    int action_visits = root->children[a].num_visits;
    std::unique_ptr<SN> new_root = std::move(root->children[a].children[s]);
    int state_visits = new_root->num_visits;
    step_risk_thd *= action_visits / (static_cast<float>(state_visits) + 0.0001f);
    root = std::move(new_root);
    root->parent = nullptr;
    for (auto& child : root->children) {
        child.parent = root.get();
    }
}

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
void tree_search<S, A, SN, AN>::play() {
    spdlog::debug("Running simulations");
    for (int i = 0; i < num_sim; i++) {
        spdlog::trace("Simulation " + std::to_string(i));
        SN* leaf = select();
        expand(leaf);
        propagate(leaf);
    }

    A a = root->select_action(risk_thd, false);

    spdlog::trace("Play action: " + std::to_string(a));
    auto [s, r, p, e] = agent<S, A>::handler.play_action(a);
    spdlog::trace("  Result: s=" + std::to_string(s) + ", r=" + std::to_string(r) + ", p=" + std::to_string(p));
    
    root->children[a].add_outcome(s, r, p, e);

    descent(a, s);
}

} // namespace ts
} // namespace gym
