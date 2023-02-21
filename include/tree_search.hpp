#pragma once

#include "world.hpp"

#include <vector>
#include <map>

namespace world {
namespace st {


template <typename S, typename A, typename N_DATA>
struct node {
    node *parent = nullptr;
    A const* incoming_action = nullptr;
    S const* state = nullptr;

    float reward = 0;
    float penalty = 0;

    // predicted expected future reward and penalty
    float future_reward = 0;
    float future_penalty = 0;

    int num_visits = 0;

    N_DATA data;
    std::map<std::pair<A, S>, node> children;

    bool is_leaf() const {
        return children.empty();
    }

    bool is_root() const {
        return parent == nullptr;
    }
};

/*********************************************************************
 * @brief Search tree
 * 
 * @tparam S type of states
 * @tparam A type of actions
 */
template <typename S, typename A, typename N_DATA>
class search_tree : public agent<S, A> {
public:
    using node_t = node<S, A, N_DATA>;
    using env_t = environment<S, A>;
protected:
    int max_depth;
    int num_sim;

    // discount factor
    float gamma;

    node_t root;
    environment<S, A> *env;
    A const* last_action;
    std::vector<node_t> history;

    /**
     * @brief Simulate a game from the current root node till a leaf node or max_depth is reached
     * 
     */
    void simulate();

    /**
     * @brief Backpropagate the result of a simulation from the leaf node to the root
     * 
     * @param nod
     */
    void backprop(node_t *nod);

    /**
     * @brief Backpropagate the result of a simulation from a child node to its parent
     * 
     * @param child
     * @param val
     * @param reg
     */
    virtual void _backprop(node_t *child, float val, float reg);

    /**
     * @brief Get the best action from a given node
     * 
     * @return const A& 
     */
    virtual const A& select_action(node_t *nod) = 0;

    /**
     * @brief Get an explorative action a given node used during the simulation
     * 
     * @return const A& 
     */
    virtual const A& explore_action(node_t *nod) = 0;

    /**
     * @brief Expand a leaf node
     * 
     * @param nod 
     */
    void expand(node_t *nod);
    
    /**
     * @brief Initialize a successor node
     * 
     * @param parent
     * @param state
     * @param action
     */
    virtual void init_node(node_t *parent, const S& state, const A& action) = 0;

    void prune(const S& state);
public:
    /* Constructors */
    search_tree(env_t* env);
    virtual ~search_tree() = default;


    /* Agent interface */
    const A& get_action() override;
    void pass_outcome(outcome_t<S> outcome) override;
    virtual void reset() override = 0;
};

} // namespace st
} // namespace world

#include "tree_search.ipp"
