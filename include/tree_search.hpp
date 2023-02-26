#pragma once

#include "world.hpp"

#include <vector>
#include <map>

namespace world {
namespace ts {

/*********************************************************************
 * @brief Search tree
 * 
 * @tparam S type of states
 * @tparam SN type of state nodes
 * @tparam AN type of action nodes
 */
template <typename S, template <typename> class SN, template <typename> class AN>
class tree_search : public agent<S> {
public:
    using state_node_t = SN<S>;
    using action_node_t = AN<S>;
    using env_t = environment<S>;
    using handler_t = environment_handler<S>;
protected:
    int max_depth;
    int num_sim;
    float risk_thd; // risk threshold
    float gamma; // discount factor

    state_node_t root;

    /**
     * @brief Select a leaf node to expand
     * 
     * @return state_node_t*
     */
    state_node_t* select();

    /**
     * @brief Expand a leaf node
     * 
     * @param nod 
     */
    void expand(state_node_t* leaf);

    /**
     * @brief Propagate the result of a simulation from the leaf node back to the root
     * 
     * @param nod
     */
    void propagate(state_node_t* leaf);

    /**
     * @brief Prune the search tree by making a step from the root to state `s` along action `a`. S is a new root of the tree
     * 
     * @param a Action taken
     * @param s State reached
     */
    void descent(action_t a, S s);

public:
    /**
     * @brief Construct a new search tree object
     * 
     * @param max_depth Maximum depth of the search tree
     * @param num_sim Number of simulations to run at each leaf node
     * @param gamma Discount factor
     */
    tree_search(int max_depth, int num_sim, float risk_thd, float gamma)
    : agent<S>(), max_depth(max_depth), num_sim(num_sim), risk_thd(risk_thd), gamma(gamma) {
        root.parent = nullptr;
    }

    void play() override;

    std::string name() const override {
        return "Tree Search";
    }
};

} // namespace st
} // namespace world

#include "tree_search.ipp"
