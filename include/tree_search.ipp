#pragma once

#include "world.hpp"

#include <vector>
#include <map>
#include <cassert>

namespace world {
namespace ts {

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
SN* tree_search<S, A, SN, AN>::select() {
    state_node_t* current_state = root.get();

    int depth = 0;

    float run_risk_thd = step_risk_thd;

    while (!current_state->is_leaf() && depth < max_depth && !current_state->is_terminal) {
        A action = current_state->select_action(run_risk_thd, true);
        action_node_t *current_action = &current_state->children[action];
        auto [s, r, p, t] = agent<S, A>::handler.sim_action(action);
        current_action->add_outcome(s, r, p, t);
        current_state = current_action->children[s].get();

        int state_visits = current_state->num_visits;
        int action_visits = current_action->num_visits;

        run_risk_thd *= action_visits / static_cast<float>(state_visits + 0.0001);

        depth++;
    }

    agent<S, A>::handler.sim_reset();
    return current_state;
}

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
void tree_search<S, A, SN, AN>::expand(SN* leaf) {
    leaf->expand(agent<S, A>::handler.num_actions());
}

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
void tree_search<S, A, SN, AN>::propagate(state_node_t* leaf) {
    action_node_t* prev_action = nullptr;
    state_node_t* current_state = leaf;

    while (!current_state->is_root()) {
        current_state->propagate(prev_action, gamma);
        action_node_t* current_action = current_state->parent;
        current_action->propagate(current_state, gamma);
        current_state = current_action->parent;
        prev_action = current_action;
    }

    current_state->propagate(prev_action, gamma);
}

template <typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
void tree_search<S, A, SN, AN>::descent(A a, S s) {
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
        spdlog::debug("Simulation " + std::to_string(i));
        SN* leaf = select();
        expand(leaf);
        propagate(leaf);
    }

    A a = root->select_action(risk_thd, false);

    spdlog::debug("Play action: " + std::to_string(a));
    auto [s, r, p, e] = agent<S, A>::handler.play_action(a);
    spdlog::debug("  Result: s=" + std::to_string(s) + ", r=" + std::to_string(r) + ", p=" + std::to_string(p));
    
    root->children[a].add_outcome(s, r, p, e);

    descent(a, s);
}

} // namespace ts
} // namespace world
