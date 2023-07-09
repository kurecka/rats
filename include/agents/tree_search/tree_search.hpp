#pragma once

#include "utils.hpp"
#include <string>
#include <vector>

namespace gym {
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
        // descend_callback(sn, action, an, s, new_sn);
        depth++;
        sn = new_sn;
    }
    return sn;
}

template<
typename S, typename A, typename DATA, typename V, typename Q,
    void (*prop_v)(state_node<S, A, DATA, V, Q>*, action_node<S, A, DATA, V, Q>*, float),
    void (*prop_q)(action_node<S, A, DATA, V, Q>*, state_node<S, A, DATA, V, Q>*, float)
>
void propagate(state_node<S, A, DATA, V, Q>* leaf, float gamma) {
    using state_node_t = state_node<S, A, DATA, V, Q>;
    using action_node_t = action_node<S, A, DATA, V, Q>;

    action_node_t* prev_an = nullptr;
    state_node_t* current_sn = leaf;

    while (!current_sn->is_root()) {
        prop_v(current_sn, prev_an, gamma);
        action_node_t* current_an = current_sn->get_parent();
        prop_q(current_an, current_sn, gamma);
        current_sn = current_an->get_parent();
        prev_an = current_an;
    }

    prop_v(current_sn, prev_an, gamma);
}

/*********************************************************************
 * SPECIFIC FUNCTIONS
 *********************************************************************/

// TODO: implement
// template<typename S, typename A, typename DATA, typename V, typename Q>
// float rollout(state_node<S, A, DATA, V, Q>* node);

using point_value = std::pair<float, float>;

template<typename S, typename A, typename DATA, typename V, bool deterministic>
A select_action_primal(state_node<S, A, DATA, V, point_value>* node, bool explore) {
    float risk_thd = node->common_data->sample_risk_thd;
    float c = node->common_data->exploration_constant;

    auto& children = node->children;
    auto q = children[0].q;
    auto [min_r, min_p] = q;
    auto [max_r, max_p] = q;
    for (size_t i = 0; i < children.size(); ++i) {
        auto [er, ep] = children[i].q;
        min_r = std::min(min_r, er);
        max_r = std::max(max_r, er);
        min_p = std::min(min_p, ep);
        max_p = std::max(max_p, ep);
    }
    if (min_r >= max_r) max_r = min_r + 0.1f;
    if (min_p >= max_p) max_p = min_p + 0.1f;

    std::vector<float> ucts(children.size());
    std::vector<float> lcts(children.size());

    for (size_t i = 0; i < children.size(); ++i) {
        ucts[i] = children[i].q.first + explore * c * (max_r - min_r) * static_cast<float>(
            sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
        );
        lcts[i] = children[i].q.second - explore * c * (max_p - min_p) * static_cast<float>(
            sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
        );
        if (lcts[i] < 0) lcts[i] = 0;
    }

    auto [a1, p2, a2] = greedy_mix(ucts, lcts, risk_thd);
    if (!explore) {
        std::string ucts_str = "";
        for (auto u : ucts) ucts_str += std::to_string(u) + ", ";
        std::string lcts_str = "";
        for (auto l : lcts) lcts_str += std::to_string(l) + ", ";
        spdlog::trace("ucts: {}", ucts_str);
        spdlog::trace("lcts: {}", lcts_str);
        spdlog::trace("a1: {}, p2: {}, a2: {}, thd: {}", a1, p2, a2, risk_thd);
    }

    if constexpr (deterministic) {
        return node->actions[a1];
    } else {
        if (rng::unif_float() < p2) {
            return node->actions[a2];
        } else {
            return node->actions[a1];
        }
    }
}

template<typename S, typename A, typename DATA, typename V>
A select_action_dual(state_node<S, A, DATA, V, point_value>* node, bool explore) {
    float risk_thd = node->common_data->sample_risk_thd;
    float lambda = node->common_data->lambda;
    float c = node->common_data->exploration_constant;

    auto& children = node->children;
    
    float min_v = children[0].q.first - lambda * children[0].q.second;
    float max_v = min_v;
    std::vector<float> uct_values(children.size());
    for (size_t i = 0; i < children.size(); ++i) {
        float val = children[i].q.first - lambda * children[i].q.second;
        min_v = std::min(min_v, val);
        max_v = std::max(max_v, val);
        uct_values[i] = val;
    }
    if (min_v >= max_v) max_v = min_v + 1;

    for (size_t i = 0; i < children.size(); ++i) {
        uct_values[i] += explore * c * (max_v - min_v) * static_cast<float>(
            sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
        );
    }

    float best_reward = *std::max_element(uct_values.begin(), uct_values.end());
    float eps = (max_v - min_v) * 0.1f;
    std::vector<size_t> eps_best_actions;
    for (size_t i = 0; i < children.size(); ++i) {
        if (uct_values[i] >= best_reward - eps) {
            eps_best_actions.push_back(i);
        }
    }
    
    std::vector<float> rs(eps_best_actions.size());
    for (size_t idx : eps_best_actions) {
        rs[idx] = children[idx].q.first;
    }

    std::vector<float> ps(eps_best_actions.size());
    for (size_t idx : eps_best_actions) {
        ps[idx] = children[idx].q.second;
    }

    auto [a1, p2, a2] = greedy_mix(rs, ps, risk_thd);
    a1 = eps_best_actions[a1];
    a2 = eps_best_actions[a2];
    if (rng::unif_float() < p2) {
        return node->actions[a2];
    } else {
        return node->actions[a1];
    }
}

template<typename S, typename A, typename DATA, typename V, typename Q>
void descend_callback_void(
    state_node<S, A, DATA, V, Q>*,
    A, action_node<S, A, DATA, V, Q>*,
    S, state_node<S, A, DATA, V, Q>*
) {}

template<typename S, typename A, typename DATA>
void prop_v_value(
    state_node<S, A, DATA, point_value, point_value>* sn,
    action_node<S, A, DATA, point_value, point_value>* child,
    float gamma
) {
    if (child) {
        sn->num_visits++;
        sn->v.first += (gamma * child->q.first - sn->v.first) / sn->num_visits;
        sn->v.second += (gamma * child->q.second - sn->v.second) / sn->num_visits;
    }
}


template<typename S, typename A, typename DATA>
void prop_q_value(
    action_node<S, A, DATA, point_value, point_value>* an,
    state_node<S, A, DATA, point_value, point_value>* child,
    float gamma
) {
    an->num_visits++;
    an->q.first += (gamma * (child->v.first + child->observed_reward) - an->q.first) / an->num_visits;
    an->q.second += (gamma * (child->v.second + child->observed_penalty) - an->q.second) / an->num_visits;
}

} // namespace ts
} // namespace gym
