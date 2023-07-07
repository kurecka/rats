#pragma once
#include "tree_search.hpp"
#include <string>
#include <vector>

namespace gym {
namespace ts {

/*********************************************************************
 * NODE INTERFACE
 *********************************************************************/

/**
 * @brief UCT state node
 * 
 * @tparam S State type
 */
template<typename S, typename A, typename AN, typename DATA>
struct uct_state : public general_state_node<S, A, AN, DATA> {
public:
    float expected_reward = 0;
    float expected_penalty = 0;
public:
    virtual A select_action(bool explore) = 0;
    void propagate(uct_action<S, A, DATA, MODE>* child, float gamma);
    void descend_update(A a, S s, bool is_sim);
    std::string to_string() const override;
};


/**
 * @brief Action node
 * 
 * @tparam S State type
 */
template<typename S, typename A, typename DATA, int MODE>
struct uct_action : public general_action_node<S, A, SN, DATA> {
public:
    float expected_reward = 0;
    float expected_penalty = 0;

public:
    void add_outcome(S s, float r, float p, bool t) override;
    void propagate(uct_state<S, A, DATA, MODE>* child, float gamma) override;
    uct_state<S, A, DATA, MODE>* get_child(S s) {return children[s].get();}
    std::string to_string() const override;
};


} // namespace ts
} // namespace gym

#include "uct.ipp"
