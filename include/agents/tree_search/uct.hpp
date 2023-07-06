#pragma once
#include "tree_search.hpp"
#include <string>
#include <vector>

namespace gym {
namespace ts {

enum uct_mode {
    PRIMAL = 0,
    DUAL = 1,
};

/*********************************************************************
 * NODE INTERFACE
 *********************************************************************/

template<typename DATA, int MODE>
concept CompatibleDataMode = requires (DATA d) {
    d.risk_thd;
    d.sample_risk_thd;
    d.exploration_constant;
} &&
(
    MODE == PRIMAL ||
    (MODE == DUAL && requires (DATA d) {d.lambda;})
);

template<typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
struct uct_action;

/**
 * @brief UCT state node
 * 
 * @tparam S State type
 */
template<typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
struct uct_state {
public:
    uct_action<S, A, DATA, MODE> *parent;
    std::vector<uct_action<S, A, DATA, MODE>> children;
    std::vector<A> actions;
    
    float observed_reward = 0;
    float observed_penalty = 0;
    float expected_reward = 0;
    float expected_penalty = 0;
    bool terminal = false;
    size_t num_visits = 0;

    DATA* common_data;

    A select_action_primal(bool explore);
    A select_action_dual(bool explore);
public:
    void expand(DATA* _common_data);
    A select_action(bool explore);
    void propagate(uct_action<S, A, DATA, MODE>* child, float gamma);
    uct_action<S, A, DATA, MODE>* get_child(A a) {return &children[a];}
    uct_action<S, A, DATA, MODE>*& get_parent() {return parent;}
    void descend_update(A a, S s, bool is_sim);

    size_t get_num_visits() const {return num_visits;}
    bool is_terminal() const {return terminal;}
    std::string to_string() const;
};


/**
 * @brief Action node
 * 
 * @tparam S State type
 */
template<typename S, typename A, typename DATA, int MODE>
requires CompatibleDataMode<DATA, MODE>
struct uct_action {
public:
    uct_state<S, A, DATA, MODE> *parent;
    std::map<S, std::unique_ptr<uct_state<S, A, DATA, MODE>>> children;
    
    float expected_reward = 0;
    float expected_penalty = 0;

    size_t num_visits = 0;

    DATA* common_data;

public:
    void add_outcome(S s, float r, float p, bool t);
    void propagate(uct_state<S, A, DATA, MODE>* child, float gamma);
    uct_state<S, A, DATA, MODE>* get_child(S s) {return children[s].get();}
    std::unique_ptr<uct_state<S, A, DATA, MODE>>&& get_child_unique_ptr(S s) {return std::move(children[s]);}
    uct_state<S, A, DATA, MODE>*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
    std::string to_string() const;
};


} // namespace ts
} // namespace gym

#include "uct.ipp"
