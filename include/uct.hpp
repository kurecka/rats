#pragma once
#include "tree_search.hpp"
#include <string>
#include <vector>

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


/**
 * @brief Action node
 * 
 * @tparam S 
 */
template<typename S>
struct action_node {
public:
    state_node<S> *parent;
    std::map<S, std::unique_ptr<state_node<S>>> children;
    
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

/*********************************************************************
 * @brief Simple tree search
 * 
 * @tparam S 
 *********************************************************************/

template <typename S>
class UCT : public tree_search<S, state_node, action_node> {
public:
    UCT(int max_depth, int num_sim, float risk_thd, float gamma)
    : tree_search<S, state_node, action_node>(max_depth, num_sim, risk_thd, gamma) {}

    std::string name() const override {
        return "UCT";
    }
};

} // namespace ts
} // namespace world

#include "uct.ipp"