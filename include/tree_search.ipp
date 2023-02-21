#pragma once

#include "world.hpp"

#include <vector>
#include <map>

namespace world {
namespace st {

template <typename S, typename A, typename N_DATA>
void search_tree<S, A, N_DATA>::simulate() {
    env->restore_checkpoint();

    int depth = 0;
    node_t *current = &root;
    
    float reward;
    float penalty;
    while (!current->is_leaf()) {
        const A& action = explore_action(current);
        const auto& [s, g, p, e] = env->play_action(action);
        reward = g;
        penalty = p;
        current = &current->children[{action, s}];
        depth++;
    }

    current->reward = reward;
    current->penalty = penalty;

    if (depth < max_depth && !env->is_over()) {
        expand(current);
    }

    backprop(current);
}

template <typename S, typename A, typename N_DATA>
void search_tree<S, A, N_DATA>::backprop(node_t *nod) {
    ++(nod->num_visits);

    float val = nod->future_reward;
    float reg = nod->future_penalty;

    while (!nod->is_root()) {
        val = nod->reward + gamma * val;
        reg = nod->penalty + reg;
        A const* a = nod->incoming_action;

        _backprop(nod, val, reg);
        
        nod = nod->parent;
    }
}

template <typename S, typename A, typename N_DATA>
void search_tree<S, A, N_DATA>::_backprop(node_t *nod, float val, float reg) {
    node_t* parent = nod->parent;
    ++(parent->num_visits);
    auto& a_data = parent->data.action_data[*(nod->incoming_action)];
    ++(a_data.num_visits);
    a_data.mean_payoff += (val - a_data.average_payoff) / a_data.num_visits;
    a_data.mean_penalty += (reg - a_data.average_penalty) / a_data.num_visits;
}

template <typename S, typename A, typename N_DATA>
void search_tree<S, A, N_DATA>::expand(node<S, A, N_DATA> *nod) {
    const auto& action_space = env->get_action_space();
    for (const auto& a : action_space) {
        for (const auto& s : env->get_next_states(*nod->state, a)) {
            init_node(nod, a, s);
        }
    }
}

template <typename S, typename A, typename N_DATA>
void search_tree<S, A, N_DATA>::prune(const S& state) {
    history.push_back(std::move(root));
    root = std::move(history.back().children[{*last_action, state}]);
    root.parent = nullptr;
    history.back().children.clear();
}


template <typename S, typename A, typename N_DATA>
search_tree<S, A, N_DATA>::search_tree(environment<S, A>* env) : env(env) {
    reset();
}


/***********************************************************
 * AGENT INTERFACE
 * *********************************************************/
template <typename S, typename A, typename N_DATA>
const A& search_tree<S, A, N_DATA>::get_action() {
    for (int i = 0; i < num_sim; i++) {
        simulate();
    }
    last_action = select_action(&root);
    return last_action;
}

template <typename S, typename A, typename N_DATA>
void search_tree<S, A, N_DATA>::pass_outcome(outcome_t<S> outcome) {
    const S& new_state = std::get<0>(outcome);
    prune(new_state);
}

} // namespace st
} // namespace world