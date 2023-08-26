#pragma once

#include "utils.hpp"
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

namespace rats {
namespace ts {

/*********************************************************************
 * NODE INTERFACE
 *********************************************************************/

template<typename S, typename A, typename DATA, typename V, typename Q>
struct action_node;

/**
 * @brief Tree search state node
 * 
 * @tparam S State type
 * @tparam A Action type
 * @tparam DATA Common data type
 * @tparam V V value type
 * @tparam Q Q value type
 */
template<typename S, typename A, typename DATA, typename V, typename Q>
struct state_node {
    using action_node_t = action_node<S, A, DATA, V, Q>;
public:
    S state;
    action_node_t *parent;
    std::vector<action_node_t> children;
    std::vector<A> actions;
    
    float observed_reward = 0;
    float observed_penalty = 0;
    float rollout_reward = 0;
    float rollout_penalty = 0;
    bool terminal = false;
    size_t num_visits = 0;

    V v;
    DATA* common_data;
public:
    action_node_t* get_child(A a) {return &children[a];}
    action_node_t*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
    bool is_terminal() const {return terminal;}
    bool is_root() const {return parent == nullptr;}
    bool is_leaf() const {return children.empty();}
};


/**
 * @brief Tree search action node
 * 
 * @tparam S State type
 * @tparam A Action type
 * @tparam DATA Common data type
 * @tparam V V value type
 * @tparam Q Q value type
 */
template<typename S, typename A, typename DATA, typename V, typename Q>
struct action_node {
    using state_node_t = state_node<S, A, DATA, V, Q>;
public:
    A action;
    state_node_t *parent;
    std::map<S, std::unique_ptr<state_node_t>> children;

    size_t num_visits = 0;

    float rollout_reward = 0;
    float rollout_penalty = 0;

    DATA* common_data;
    Q q;

public:
    state_node_t* get_child(S s) {return children[s].get();}
    std::unique_ptr<state_node_t>&& get_child_unique_ptr(S s) {return std::move(children[s]);}
    state_node_t*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
};


/*********************************************************************
 * TREE SEARCH
 *********************************************************************/
template<typename S, typename A, typename DATA, typename V, typename Q>
// std::unique_ptr<state_node<S, A, DATA, V, Q>>
void expand_state(state_node<S, A, DATA, V, Q>* sn) {
    sn->actions = sn->common_data->handler.possible_actions();
    sn->children.resize(sn->actions.size());
    for (size_t i = 0; i < sn->actions.size(); ++i) {
        sn->children[i].action = sn->actions[i];
        sn->children[i].parent = sn;
        sn->children[i].common_data = sn->common_data;
    }
}

template<typename S, typename A, typename DATA, typename V, typename Q>
std::unique_ptr<state_node<S, A, DATA, V, Q>> expand_action(
    action_node<S, A, DATA, V, Q>* an,
    S s, float r, float p, bool t
) {
    using state_node_t = state_node<S, A, DATA, V, Q>;

    std::unique_ptr<state_node_t> new_sn = std::make_unique<state_node_t>();
    new_sn->state = s;
    new_sn->parent = an;
    new_sn->common_data = an->common_data;
    new_sn->observed_reward = r;
    new_sn->observed_penalty = p;
    new_sn->terminal = t;

    return new_sn;
}

template<typename S, typename A, typename DATA, typename V, typename Q,
    A (*select)(state_node<S, A, DATA, V, Q>*, bool),
    void (*descent_callback)(state_node<S, A, DATA, V, Q>*, A, action_node<S, A, DATA, V, Q>*, S, state_node<S, A, DATA, V, Q>*)
>
state_node<S, A, DATA, V, Q>* select_leaf(
    state_node<S, A, DATA, V, Q>* root, bool explore = true, int max_depth = 10
) {
    using state_node_t = state_node<S, A, DATA, V, Q>;
    using action_node_t = action_node<S, A, DATA, V, Q>;

    state_node_t* sn = root;

    int depth = 0;

    while (!sn->is_leaf() && depth < max_depth && !sn->is_terminal()) {
        A action = select(sn, explore);
        action_node_t *an = sn->get_child(action);
        auto [s, r, p, t] = sn->common_data->handler.sim_action(action);
        if (an->children.find(s) == an->children.end()) {
            an->children[s] = expand_action(an, s, r, p, t);
        }
        state_node_t* new_sn = an->get_child(s);
        descent_callback(sn, action, an, s, new_sn);
        depth++;
        sn = new_sn;
    }
    return sn;
}

template<
typename S, typename A, typename DATA, typename V, typename Q,
    void (*prop_v)(state_node<S, A, DATA, V, Q>*, float, float),
    void (*prop_q)(action_node<S, A, DATA, V, Q>*, float, float)
>
void propagate(state_node<S, A, DATA, V, Q>* leaf, float gamma) {
    using state_node_t = state_node<S, A, DATA, V, Q>;
    using action_node_t = action_node<S, A, DATA, V, Q>;

    action_node_t* prev_an = nullptr;
    state_node_t* current_sn = leaf;

    float disc_r = leaf->rollout_reward;
    float disc_p = leaf->rollout_penalty;

    while (!current_sn->is_root()) {
        prop_v(current_sn,  disc_r, disc_p);
        disc_r = current_sn->observed_reward + gamma * disc_r;
        disc_p = current_sn->observed_penalty + gamma * disc_p;
        action_node_t* current_an = current_sn->get_parent();
        prop_q(current_an, disc_r, disc_p);
        current_sn = current_an->get_parent();
        prev_an = current_an;
    }

    prop_v(current_sn, disc_r, disc_p);
}

/*********************************************************************
 * SPECIFIC FUNCTIONS
 *********************************************************************/

using point_value = std::pair<float, float>;


template<typename S, typename A, typename DATA, typename V, typename Q>
void void_descend_callback(
    state_node<S, A, DATA, V, Q>*,
    A, action_node<S, A, DATA, V, Q>*,
    S, state_node<S, A, DATA, V, Q>*
) {}

template<typename S, typename A, typename DATA, typename V, typename Q>
void void_rollout(state_node<S, A, DATA, V, Q>*) {}

template<typename S, typename A, typename DATA>
void uct_prop_v_value(
    state_node<S, A, DATA, point_value, point_value>* sn,
    float disc_r, float disc_p
) {
    sn->num_visits++;
    sn->v.first += (disc_r - sn->v.first) / sn->num_visits;
    sn->v.second += (disc_p - sn->v.second) / sn->num_visits;
}


template<typename S, typename A, typename DATA>
void uct_prop_q_value(
    action_node<S, A, DATA, point_value, point_value>* an,
    float disc_r, float disc_p
) {
    an->num_visits++;
    an->q.first += (disc_r - an->q.first) / an->num_visits;
    an->q.second += (disc_p - an->q.second) / an->num_visits;
}

} // namespace ts
} // namespace rats
