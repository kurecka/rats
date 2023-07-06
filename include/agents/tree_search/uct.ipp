#include "utils.hpp"

namespace gym {
namespace ts {

/***************************************************
 * State node implementation
 * *************************************************/

template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
std::string uct_state<S, A, DATA, MODE>::to_string() const {
    std::string s = "State node: ";
    s += "R: " + std::to_string(observed_reward) + " P: " + std::to_string(observed_penalty) + "\\n";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + "\\n";
    s += "V: " + std::to_string(num_visits) + "\\n";
    s += "T: " + std::to_string(terminal);
    return s;
}

template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
void uct_state<S, A, DATA, MODE>::expand(DATA* _common_data) {
    common_data = _common_data;
    actions = common_data->handler.possible_actions();
    children.resize(actions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
        children[i].parent = this;
        children[i].common_data = common_data;
    }
}

template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
A uct_state<S, A, DATA, MODE>::select_action_primal(bool explore) {
    float risk_thd = common_data->sample_risk_thd;
    float c = common_data->exploration_constant;

    float min_r = children[0].expected_reward;
    float max_r = min_r;
    float min_p = children[0].expected_penalty;
    float max_p = min_p;
    for (size_t i = 0; i < children.size(); ++i) {
        min_r = std::min(min_r, children[i].expected_reward);
        max_r = std::max(max_r, children[i].expected_reward);
        min_p = std::min(min_p, children[i].expected_penalty);
        max_p = std::max(max_p, children[i].expected_penalty);
    }
    if (min_r >= max_r) max_r = min_r + 0.1f;
    if (min_p >= max_p) max_p = min_p + 0.1f;

    std::vector<float> ucts(children.size());
    std::vector<float> lcts(children.size());

    for (size_t i = 0; i < children.size(); ++i) {
        ucts[i] = children[i].expected_reward + explore * c * (max_r - min_r) * static_cast<float>(
            sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001))
        );
        lcts[i] = children[i].expected_penalty - explore * c * (max_p - min_p) * static_cast<float>(
            sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001))
        );
        if (lcts[i] < 0) lcts[i] = 0;
    }

    auto [a1, p2, a2] = greedy_mix(ucts, lcts, risk_thd);
    // if (!explore){
        // std::string s1 = "";
        // for (float u : ucts) s1 += std::to_string(u) + " ";
        // spdlog::info("ucts: {}", s1);
        // std::string s2 = "";
        // for (float l : lcts) s2 += std::to_string(l) + " ";
        // spdlog::info("lcts: {}", s2);
        // spdlog::info("a1: {}, p2: {}, a2: {}, thd: {}", a1, p2, a2, risk_thd);
    // }
    if (MODE == DPRIMAL) {
        return actions[a1];
    } else {
        if (rng::unif_float() < p2) {
            return actions[a2];
        } else {
            return actions[a1];
        }
    }
}


template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
A uct_state<S, A, DATA, MODE>::select_action_dual(bool explore) {
    float lambda = common_data->lambda;
    float c = common_data->exploration_constant;
    
    float min_v = children[0].expected_reward - lambda * children[0].expected_penalty;
    float max_v = min_v;
    std::vector<float> uct_values(children.size());
    for (size_t i = 0; i < children.size(); ++i) {
        float val = children[i].expected_reward - lambda * children[i].expected_penalty;
        min_v = std::min(min_v, val);
        max_v = std::max(max_v, val);
        uct_values[i] = val;
    }
    if (min_v >= max_v) max_v = min_v + 1;

    for (size_t i = 0; i < children.size(); ++i) {
        uct_values[i] += explore * c * (max_v - min_v) * static_cast<float>(
            sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001))
        );
    }

    size_t best_action = 0;
    float best_reward = uct_values[0];
    for (size_t i = 1; i < children.size(); ++i) {
        if (uct_values[i] > best_reward) {
            best_reward = uct_values[i];
            best_action = i;
        }
    }

    return actions[best_action];
}


template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
A uct_state<S, A, DATA, MODE>::select_action(bool explore) {
    if constexpr (MODE == PRIMAL || MODE == DPRIMAL) {
        return select_action_primal(explore);
    } else if constexpr (MODE == DUAL || MODE == DDUAL) {
        return select_action_dual(explore);
    }
}


template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
void uct_state<S, A, DATA, MODE>::propagate(uct_action<S, A, DATA, MODE>* child, float gamma) {
    if (child) {
        num_visits++;
        expected_reward += (gamma * child->expected_reward - expected_reward) / num_visits;
        expected_penalty += (gamma * child->expected_penalty - expected_penalty) / num_visits;
    }
}


template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
void uct_state<S, A, DATA, MODE>::descend_update(A, S, bool) {
    // if (MODE == DUAL && is_sim) return;
    // uct_action<S, A, DATA, MODE>* descendant_action_node = get_child(a);
    // uct_state<S, A, DATA, MODE>* descendant_state_node = descendant_action_node->get_child(s);
    // size_t action_visits = descendant_action_node->get_num_visits();
    // size_t state_visits = descendant_state_node->get_num_visits();

    // if (is_sim) {
    //     common_data->sample_risk_thd *= action_visits / static_cast<float>(state_visits + 0.0001);
    // } else {
    //     common_data->risk_thd *= action_visits / static_cast<float>(state_visits + 0.0001);
    // }
}

/***************************************************
 * Action node implementation
 * *************************************************/

template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
void uct_action<S, A, DATA, MODE>::add_outcome(S s, float r, float p, bool t) {
    if (children.find(s) == children.end()) {
        children[s] = std::make_unique<uct_state<S, A, DATA, MODE>>();
    }
    children[s]->observed_reward = r;
    children[s]->observed_penalty = p;

    children[s]->parent = this;
    children[s]->terminal = t;
}

template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
void uct_action<S, A, DATA, MODE>::propagate(uct_state<S, A, DATA, MODE>* child, float gamma) {
    num_visits++;
    expected_reward += (gamma * (child->expected_reward + child->observed_reward) - expected_reward) / num_visits;
    expected_penalty += (gamma * (child->expected_penalty + child->observed_penalty) - expected_penalty) / num_visits;
}

template <typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
std::string uct_action<S, A, DATA, MODE>::to_string() const {
    std::string s = "Action node: ";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + " ";
    s += "V: " + std::to_string(num_visits) + " ";
    return s;
}

} // namespace ts
} // namespace gym
