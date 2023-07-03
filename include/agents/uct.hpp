#pragma once
#include "tree_search.hpp"
#include <string>
#include <vector>

namespace gym {
namespace ts {

/*********************************************************************
 * NODE INTERFACE
 *********************************************************************/
template<typename S, typename A>
struct action_node;


/**
 * @brief State node
 * 
 * @tparam S State type
 */
template<typename S, typename A>
struct state_node {
public:
    action_node<S, A> *parent;
    std::vector<action_node<S, A>> children;
    
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

    void expand(size_t num_actions);
    A select_action(float risk_thd, bool explore);
    void propagate(action_node<S, A>* child, float gamma);

public:
    void validate() const;
};


/**
 * @brief Action node
 * 
 * @tparam S State type
 */
template<typename S, typename A>
struct action_node {
public:
    state_node<S, A> *parent;
    std::map<S, std::unique_ptr<state_node<S, A>>> children;
    
    float expected_reward = 0;
    float expected_penalty = 0;

    int num_visits = 0;

public:
    void add_outcome(S s, float r, float p, bool t);
    void propagate(state_node<S, A>* child, float gamma);
    std::string to_string() const;

public:
    void validate() const;
};

/*********************************************************************
 * @brief Simple tree search
 * 
 * @tparam S State type
 *********************************************************************/

template <typename S, typename A>
class UCT : public tree_search<S, A, state_node<S, A>, action_node<S, A>> {
public:
    UCT(int _max_depth, int _num_sim, float _risk_thd, float _gamma)
    : tree_search<S, A, state_node<S, A>, action_node<S, A>>(_max_depth, _num_sim, _risk_thd, _gamma) {}

    std::string name() const override {
        return "UCT";
    }
};

} // namespace ts
} // namespace gym

#include "uct.ipp"
