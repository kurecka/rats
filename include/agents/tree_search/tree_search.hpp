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
    std::map<S, size_t> child_idx;

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
void expand_action(
    action_node<S, A, DATA, V, Q>* an,
    S s, float r, float p, bool t
) {
    using state_node_t = state_node<S, A, DATA, V, Q>;
    // TODO: initialize all possible child states

    an->child_idx[s] = an->children.size();

    std::unique_ptr<state_node_t>& new_sn = an->children[s] = std::make_unique<state_node_t>();
    new_sn->state = s;
    new_sn->parent = an;
    new_sn->common_data = an->common_data;
    new_sn->observed_reward = r;
    new_sn->observed_penalty = p;
    new_sn->terminal = t;
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
            expand_action(an, s, r, p, t);
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
        disc_p = current_sn->observed_penalty + disc_p;
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

/**
 * @brief Monte carlo rollout function
 * 
 * @param sn A leaf node to rollout
 * 
 * Do a Monte Carlo rollout from the given leaf node. Update its rollout reward and penalty.
 */
template<typename S, typename A, typename DATA, typename V, typename Q>
void rollout(state_node<S, A, DATA, V, Q>* sn) {
    using state_node_t = state_node<S, A, DATA, V, Q>;

    state_node_t* current_sn = sn;
    float disc_r = 0;
    float disc_p = 0;
    float gamma_pow = 1.0;
    bool terminal = current_sn->is_terminal();

    while (!terminal) {
        gamma_pow *= current_sn->common_data->gamma;
        A action = current_sn->common_data->handler.get_action(
            rng::unif_int(current_sn->common_data->handler.num_actions())
        );
        auto [s, r, p, t] = current_sn->common_data->handler.sim_action(action);
        terminal = t;
        disc_r = r + disc_r * gamma_pow;
        disc_p = p + disc_p * gamma_pow;
    }

    current_sn->rollout_reward = disc_r;
    current_sn->rollout_penalty = disc_p;
}


/**
 * @brief Monte carlo rollout function
 * 
 * @param sn A leaf node to rollout
 * 
 * Do a Monte Carlo rollout from the given leaf node. Update its rollout reward and penalty.
 */
template<typename S, typename A, typename DATA, typename V, typename Q, A a, int N>
void constant_rollout(state_node<S, A, DATA, V, Q>* sn) {
    using state_node_t = state_node<S, A, DATA, V, Q>;

    float mean_r = 0;
    // float mean_p = 0;
    auto common_data = sn->common_data;

    common_data->handler.make_checkpoint(1);

    for (int i = 0; i < N; ++i) {
        state_node_t* current_sn = sn;
        float disc_r = 0;
        // float disc_p = 0;
        float gamma_pow = 1.0;
        bool terminal = current_sn->is_terminal();

        while (!terminal) {
            gamma_pow *= common_data->gamma;
            auto [s, r, p, t] = common_data->handler.sim_action(a);
            terminal = t;
            disc_r = r + disc_r * gamma_pow;
            // disc_p = p + disc_p * gamma_pow;
        }

        mean_r += disc_r;
        // mean_p += disc_p;
        common_data->handler.restore_checkpoint(1);
    }

    sn->rollout_reward = mean_r / N;
    float probs[] = {1.000000f,0.428571f,0.183673f,0.078717f,0.033736f,0.014458f,0.006196f,0.002656f,0.001138f,0.000488f,0.000209f,0.000090f,0.000038f,0.000016f,0.000007f,0.000003f,0.000001f,0.000001f,0.000000f,0.000000f,0.000000f};
    int s = sn->state;
    s = std::min(s, 20);
    s = std::max(s, 0);
    sn->rollout_penalty = probs[s];
}


template<typename S, typename A, typename DATA, typename V, typename Q, A a, int N>
void constant_rollout(action_node<S, A, DATA, V, Q>* an) {
    float mean_r = 0;
    float mean_p = 0;
    auto common_data = an->common_data;

    common_data->handler.make_checkpoint(1);

    for (int i = 0; i < N; ++i) {
        A initial_action = an->action;
        auto [s, r, p, t] = common_data->handler.sim_action(initial_action);
        bool terminal = t;
        float disc_r = r;
        float disc_p = p;
        float gamma_pow = 1.0;

        while (!terminal) {
            gamma_pow *= common_data->gamma;
            auto [s_, r_, p_, t_] = common_data->handler.sim_action(a);
            terminal = t_;
            disc_r = r_ + disc_r * gamma_pow;
            disc_p = p_ + disc_p * gamma_pow;
        }

        mean_r += disc_r;
        mean_p += disc_p;
        common_data->handler.restore_checkpoint(1);
    }

    an->rollout_reward = mean_r / N;
    an->rollout_penalty = mean_p / N;
}

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
