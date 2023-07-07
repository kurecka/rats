#include "utils.hpp"

namespace gym {
namespace ts {

/***************************************************
 * State node implementation
 * *************************************************/

template <typename S, typename A, typename AN, typename DATA>
std::string uct_state<S, A, AN, DATA>::to_string() const {
    std::string s = "State node: ";
    s += "R: " + std::to_string(observed_reward) + " P: " + std::to_string(observed_penalty) + "\\n";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + "\\n";
    s += "V: " + std::to_string(num_visits) + "\\n";
    s += "T: " + std::to_string(terminal);
    return s;
}


template <typename S, typename A, typename AN, typename DATA>
void uct_state<S, A, AN, DATA>::expand(DATA* _common_data) {
    common_data = _common_data;
    actions = common_data->handler.possible_actions();
    children.resize(actions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
        children[i].parent = this;
        children[i].common_data = common_data;
    }
}


template <typename S, typename A, typename AN, typename DATA>
void uct_state<S, A, AN, DATA>::propagate(uct_action<S, A, DATA, MODE>* child, float gamma) {
    if (child) {
        num_visits++;
        expected_reward += (gamma * child->expected_reward - expected_reward) / num_visits;
        expected_penalty += (gamma * child->expected_penalty - expected_penalty) / num_visits;
    }
}


template <typename S, typename A, typename AN, typename DATA>
void uct_state<S, A, AN, DATA>::descend_update(A, S, bool) {
    // if (MODE == DUAL typename AN, && is_sim

    // uct_state<S, A, AN, DATA>* descendant_state_node = descendant_action_node->get_child(s);
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

template <typename S, typename A, typename SN, typename DATA>
void uct_action<S, A, SN, DATA>::add_outcome(S s, float r, float p, bool t) {
    if (children.find(s) == children.end()) {
        children[s] = std::make_unique<SN>();
    }
    children[s]->observed_reward = r;
    children[s]->observed_penalty = p;

    children[s]->parent = this;
    children[s]->terminal = t;
}


template <typename S, typename A, typename SN, typename DATA>
void uct_action<S, A, SN, DATA>::propagate(SN* child, float gamma) {
    num_visits++;
    expected_reward += (gamma * (child->expected_reward + child->observed_reward) - expected_reward) / num_visits;
    expected_penalty += (gamma * (child->expected_penalty + child->observed_penalty) - expected_penalty) / num_visits;
}

template <typename S, typename A, typename SN, typename DATA>
std::string uct_action<S, A, SN, DATA>::to_string() const {
    std::string s = "Action node: ";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + " ";
    s += "V: " + std::to_string(num_visits) + " ";
    return s;
}

} // namespace ts
} // namespace gym
