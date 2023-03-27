#pragma once

#include "world.hpp"

#include <vector>
#include <map>
#include <cassert>

namespace world {
namespace ts {

/*********************************************************************
 * NODE INTERFACE
 *********************************************************************/
template<typename S>
struct action_node;


/**
 * @brief State node
 * 
 * @tparam S 
 */
template<typename S>
struct state_node {
public:
    action_node<S> *parent;
    std::vector<action_node<S>> children;
    
    float observed_reward = 0;
    float observed_penalty = 0;
    float expected_reward = 0;
    float expected_penalty = 0;
    bool is_terminal = false;
    int num_visits = 0;
public:
    bool is_root() const {
        return parent == nullptr;
    }

    bool is_leaf() const {
        return children.empty();
    }

    std::string to_string() const;

    void expand(int num_actions);
    action_t select_action(float risk_thd, bool explore);
    void propagate(action_node<S>* child, float gamma);

public:
    void validate() const;
};

template <typename S>
std::string state_node<S>::to_string() const {
    std::string s = "State node: ";
    s += "R: " + std::to_string(observed_reward) + " P: " + std::to_string(observed_penalty) + "\\n";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + "\\n";
    s += "V: " + std::to_string(num_visits) + "\\n";
    s += "T: " + std::to_string(is_terminal);
    return s;
}

template <typename S>
void state_node<S>::expand(int num_actions) {
    children.resize(num_actions);
    for (int i = 0; i < num_actions; ++i) {
        children[i].parent = this;
    }
}

template <typename S>
action_t state_node<S>::select_action(float risk_thd, bool explore) {
    // best safe action
    float best_reward = -1e9;
    action_t best_action = 0;
    float best_penalty = 1;
    float min_r = children[0].expected_reward;
    float max_r = min_r;
    for (size_t i = 0; i < children.size(); ++i) {
        min_r = std::min(min_r, children[i].expected_reward);
        max_r = std::max(max_r, children[i].expected_reward);
    }
    if (min_r == max_r) max_r += 1;

    for (size_t i = 0; i < children.size(); ++i) {
        float ucb = children[i].expected_reward + explore * 5 * (max_r - min_r) * sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001));
        float lcb = children[i].expected_penalty - explore * 5 * sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001));

        // logger.debug("Action: " + std::to_string(i) + " UCB: " + std::to_string(ucb) + " LCB: " + std::to_string(lcb));
        // logger.debug("Action: " + std::to_string(i) + " R: " + std::to_string(children[i].expected_reward) + " P: " + std::to_string(children[i].expected_penalty));

        if ((ucb > best_reward && lcb < best_penalty) || (best_penalty > risk_thd && lcb < best_penalty)) {
            best_reward = ucb;
            best_penalty = lcb;
            best_action = i;
        }
    }

    if (explore && unif_float() < 0.1) {
        return unif_int(0, children.size());
    } else {
        return best_action;
    }
}

template <typename S>
void state_node<S>::propagate(action_node<S>* child, float gamma) {
    if (child) {
        num_visits++;
        expected_reward += (gamma * child->expected_reward - expected_reward) / num_visits;
        expected_penalty += (gamma * child->expected_penalty - expected_penalty) / num_visits;
    }
}

template <typename S>
void state_node<S>::validate() const {

    int s = 0;
    for (const auto& c : children) {
        s += c.num_visits;
    }
    assert(s == num_visits);

    for (const auto& c : children) {
        c.validate();
    }
}

/**
 * @brief Action node
 * 
 * @tparam S 
 */
template<typename S>
struct action_node {
public:
    state_node<S> *parent;
    std::map<S, state_node<S>> children;
    
    float expected_reward = 0;
    float expected_penalty = 0;

    int num_visits = 0;

public:
    void add_outcome(S s, float r, float p, bool t);
    void propagate(state_node<S>* child, float gamma);
    std::string to_string() const;

public:
    void validate() const;
};

template <typename S>
void action_node<S>::add_outcome(S s, float r, float p, bool t) {
    children[s].observed_reward = r;
    children[s].observed_penalty = p;

    children[s].parent = this;
    children[s].is_terminal = t;
}

template <typename S>
void action_node<S>::propagate(state_node<S>* child, float gamma) {
    num_visits++;
    expected_reward += (gamma * (child->expected_reward + child->observed_reward) - expected_reward) / num_visits;
    expected_penalty += (gamma * (child->expected_penalty + child->observed_penalty) - expected_penalty) / num_visits;
}

template <typename S>
std::string action_node<S>::to_string() const {
    std::string s = "Action node: ";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + " ";
    s += "V: " + std::to_string(num_visits) + " ";
    return s;
}

template <typename S>
void action_node<S>::validate() const {
    for (const auto& [state, node] : children) {
        node.validate();
    }
}

/***********************************************************
 * AGENT INTERFACE
 * *********************************************************/

template <typename S, template <typename> class SN, template <typename> class AN>
SN<S>* tree_search<S, SN, AN>::select() {
    state_node_t* current_state = &root;

    int depth = 0;

    float run_risk_thd = step_risk_thd;

    while (!current_state->is_leaf() && depth < max_depth && !current_state->is_terminal) {
        action_t action = current_state->select_action(run_risk_thd, true);
        action_node_t *current_action = &current_state->children[action];
        auto [s, r, p, t] = agent<S>::handler.sim_action(action);
        current_action->add_outcome(s, r, p, t);
        current_state = &current_action->children[s];

        int state_visits = current_state->num_visits;
        int action_visits = current_action->num_visits;

        run_risk_thd *= action_visits / (float) (state_visits + 0.0001);

        depth++;
    }

    agent<S>::handler.sim_reset();
    return current_state;
}

template <typename S, template <typename> class SN, template <typename> class AN>
void tree_search<S, SN, AN>::expand(SN<S>* leaf) {
    leaf->expand(agent<S>::handler.num_actions());
}

template <typename S, template <typename> class SN, template <typename> class AN>
void tree_search<S, SN, AN>::propagate(state_node_t* leaf) {
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

template <typename S, template <typename> class SN, template <typename> class AN>
void tree_search<S, SN, AN>::descent(action_t a, S s) {
    int action_visits = root.children[a].num_visits;
    SN<S>& child = root.children[a].children[s];
    SN<S> new_root = std::move(child);
    int state_visits = new_root.num_visits;
    step_risk_thd *= action_visits / ((float) state_visits + 0.0001);
    root = std::move(new_root);
    root.parent = nullptr;
    for (auto& child : root.children) {
        child.parent = &root;
    }
}

template <typename S, template <typename> class SN, template <typename> class AN>
void tree_search<S, SN, AN>::play() {
    for (int i = 0; i < num_sim; i++) {
        SN<S>* leaf = select();
        expand(leaf);
        propagate(leaf);
    }

    action_t a = root.select_action(risk_thd, false);

    logger.debug("Play action: " + std::to_string(a));
    auto [s, r, p, e] = agent<S>::handler.play_action(a);
    logger.debug("  Result: s=" + std::to_string(s) + ", r=" + std::to_string(r) + ", p=" + std::to_string(p));
    
    root.children[a].add_outcome(s, r, p, e);

    descent(a, s);
}

/*********************************************************************
 * @brief Simple tree search
 * 
 * @tparam S 
 *********************************************************************/

template <typename S>
class simple_tree_search : public tree_search<S, state_node, action_node> {
public:
    simple_tree_search(int max_depth, int num_sim, float risk_thd, float gamma)
    : tree_search<S, state_node, action_node>(max_depth, num_sim, risk_thd, gamma) {}

    std::string name() const override {
        return "Simple Tree Search";
    }
};

} // namespace ts
} // namespace world