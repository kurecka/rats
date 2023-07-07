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
 * @tparam A Action type
 * @tparam AN Action node type
 * @tparam DATA Data type
 */
template<typename S, typename A, typename AN, typename DATA>
struct general_state_node {
public:
    AN *parent;
    std::vector<AN> children;
    std::vector<A> actions;
    
    float observed_reward = 0;
    float observed_penalty = 0;
    bool terminal = false;
    size_t num_visits = 0;

    DATA* common_data;
public:
    virtual void expand(DATA* _common_data) = {
        common_data = _common_data;
        actions = common_data->handler.possible_actions();
        children.resize(actions.size());
        for (size_t i = 0; i < actions.size(); ++i) {
            children[i].parent = this;
            children[i].common_data = common_data;
        }
    }
    virtual A select_action(bool explore) = 0;
    virtual void propagate(AN* child, float gamma) = 0;
    virtual void descend_update(A a, S s, bool is_sim) = 0;

    virtual std::string to_string() const = 0;

    AN* get_child(A a) {return &children[a];}
    AN*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
    bool is_terminal() const {return terminal;}
};


/**
 * @brief Action node
 * 
 * @tparam S State type
 * @tparam A Action type
 * @tparam SN State node type
 * @tparam DATA Data type
 */
template<typename S, typename A, typename SN, typename DATA>
struct uct_action {
public:
    SN *parent;
    std::map<S, std::unique_ptr<SN>> children;

    size_t num_visits = 0;

    DATA* common_data;

public:
    virtual void add_outcome(S s, float r, float p, bool t) = 0;
    virtual void propagate(SN* child, float gamma) = 0;
    
    virtual std::string to_string() const = 0;
    
    SN* get_child(S s) {return children[s].get();}
    std::unique_ptr<SN>&& get_child_unique_ptr(S s) {return std::move(children[s]);}
    SN*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
};


} // namespace ts
} // namespace gym
