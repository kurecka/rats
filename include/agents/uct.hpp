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
struct uct_action;


/**
 * @brief UCT state node
 * 
 * @tparam S State type
 */
template<typename S, typename A>
struct uct_state {
public:
    uct_action<S, A> *parent;
    std::vector<uct_action<S, A>> children;
    std::vector<A> actions;
    
    float observed_reward = 0;
    float observed_penalty = 0;
    float expected_reward = 0;
    float expected_penalty = 0;
    bool terminal = false;
    size_t num_visits = 0;
public:
    void expand(std::vector<A> _actions);
    A select_action(float risk_thd, bool explore);
    void propagate(uct_action<S, A>* child, float gamma);
    uct_action<S, A>* get_child(A a) {return &children[a];}
    uct_action<S, A>*& get_parent() {return parent;}

    size_t get_num_visits() const {return num_visits;}
    bool is_terminal() const {return terminal;}
    std::string to_string() const;
};


/**
 * @brief Action node
 * 
 * @tparam S State type
 */
template<typename S, typename A>
struct uct_action {
public:
    uct_state<S, A> *parent;
    std::map<S, std::unique_ptr<uct_state<S, A>>> children;
    
    float expected_reward = 0;
    float expected_penalty = 0;

    size_t num_visits = 0;

public:
    void add_outcome(S s, float r, float p, bool t);
    void propagate(uct_state<S, A>* child, float gamma);
    uct_state<S, A>* get_child(S s) {return children[s].get();}
    std::unique_ptr<uct_state<S, A>>&& get_child_unique_ptr(S s) {return std::move(children[s]);}
    uct_state<S, A>*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
    std::string to_string() const;
};

/*********************************************************************
 * @brief Simple tree search
 * 
 * @tparam S State type
 *********************************************************************/

template <typename S, typename A>
class UCT : public tree_search<S, A, uct_state<S, A>, uct_action<S, A>> {
public:
    UCT(int _max_depth, int _num_sim, float _risk_thd, float _gamma)
    : tree_search<S, A, uct_state<S, A>, uct_action<S, A>>(_max_depth, _num_sim, _risk_thd, _gamma) {}

    std::string name() const override {
        return "UCT";
    }
};

} // namespace ts
} // namespace gym

#include "uct.ipp"
