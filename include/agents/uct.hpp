#pragma once
#include "tree_search.hpp"
#include <string>
#include <vector>

namespace gym {
namespace ts {

/*********************************************************************
 * NODE INTERFACE
 *********************************************************************/

template<typename S, typename A, typename DATA>
struct uct_action;

/**
 * @brief UCT state node
 * 
 * @tparam S State type
 */
template<typename S, typename A, typename DATA>
struct uct_state {
public:
    uct_action<S, A, DATA> *parent;
    std::vector<uct_action<S, A, DATA>> children;
    std::vector<A> actions;
    
    float observed_reward = 0;
    float observed_penalty = 0;
    float expected_reward = 0;
    float expected_penalty = 0;
    bool terminal = false;
    size_t num_visits = 0;

    DATA* common_data;
public:
    void expand(DATA* _common_data);
    A select_action(float risk_thd, bool explore);
    void propagate(uct_action<S, A, DATA>* child, float gamma);
    uct_action<S, A, DATA>* get_child(A a) {return &children[a];}
    uct_action<S, A, DATA>*& get_parent() {return parent;}

    size_t get_num_visits() const {return num_visits;}
    bool is_terminal() const {return terminal;}
    std::string to_string() const;
};


/**
 * @brief Action node
 * 
 * @tparam S State type
 */
template<typename S, typename A, typename DATA>
struct uct_action {
public:
    uct_state<S, A, DATA> *parent;
    std::map<S, std::unique_ptr<uct_state<S, A, DATA>>> children;
    
    float expected_reward = 0;
    float expected_penalty = 0;

    size_t num_visits = 0;

    DATA* common_data;

public:
    void add_outcome(S s, float r, float p, bool t);
    void propagate(uct_state<S, A, DATA>* child, float gamma);
    uct_state<S, A, DATA>* get_child(S s) {return children[s].get();}
    std::unique_ptr<uct_state<S, A, DATA>>&& get_child_unique_ptr(S s) {return std::move(children[s]);}
    uct_state<S, A, DATA>*& get_parent() {return parent;}
    size_t get_num_visits() const {return num_visits;}
    std::string to_string() const;
};


template <typename S, typename A>
struct uct_data {
    float risk_thd;
    environment_handler<S, A>& handler;
};


/*********************************************************************
 * @brief UCT agent
 * 
 * @tparam S State type
 * @tparam A Action type
 *********************************************************************/

template <typename S, typename A>
class UCT : public agent<S, A> {
    using data_t = uct_data<S, A>;
private:
    int num_sim;
    float risk_thd;
    data_t common_data;
    tree_search<S, A, uct_state<S, A, data_t>, uct_action<S, A, data_t>, data_t> ts;
public:
    UCT(int _max_depth, int _num_sim, float _risk_thd, float _gamma)
    : agent<S, A>()
    , num_sim(_num_sim)
    , risk_thd(_risk_thd)
    , common_data({_risk_thd, agent<S, A>::handler})
    , ts(_max_depth, _risk_thd, _gamma, &common_data)
    {
        reset();
    }

    void play() override;
    void reset() override;

    std::string name() const override {
        return "UCT";
    }
};

} // namespace ts
} // namespace gym

#include "uct.ipp"
