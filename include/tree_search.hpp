#pragma once

#include "world.hpp"
#include "tree_logger.hpp"

#include <vector>
#include <map>

namespace world {
namespace ts {

/*********************************************************************
 * @brief Search tree
 * 
 * @tparam S type of states
 * @tparam SN type of state nodes
 * Should support:
 * - num_visits
 * - parent
 * - children
 * - select_action(risk_thd, explore)
 * @tparam AN type of action nodes
 * Shoul supper:
 * - num_visits
 * - children
 * - add_outcome(state, reward, penalty, terminal)
 * - propagate(state_node, gamma);
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
    float step_risk_thd; // current risk threshold during the search
    float gamma; // discount factor

    std::unique_ptr<state_node_t> root;

    /**
     * @brief Select a leaf node to expand.
     * 
     * @return state_node_t*
     */
    state_node_t* select();

    /**
     * @brief Expand a leaf node.
     * 
     * @param leaf A leaf node to be expanded
     */
    void expand(state_node_t* leaf);

    /**
     * @brief Propagate the result of a simulation from the leaf node back to the root.
     * 
     * @param leaf A leaf node whose value is to be propagated
     */
    void propagate(state_node_t* leaf);

    /**
     * @brief Prune the search tree by making a step from the root to state `s` along action `a`. S is a new root of the tree.
     * 
     * @param a Action taken.
     * @param s State reached.
     */
    void descent(action_t a, S s);

public:
    /**
     * @brief Construct a new search tree object
     * 
     * @param _max_depth Maximum depth of the search tree
     * @param _num_sim Number of simulations to run at each leaf node
     * @param _gamma Discount factor
     */
    tree_search(int _max_depth, int _num_sim, float _risk_thd, float _gamma)
    : agent<S>(), max_depth(_max_depth), num_sim(_num_sim), risk_thd(_risk_thd), gamma(_gamma) {
        reset();
    }

    void play() override;

    void reset() override {
        agent<S>::reset();

        root = std::make_unique<state_node_t>();
        root->parent = nullptr;
        step_risk_thd = risk_thd;
    }

    std::string name() const override {
        return "Tree Search";
    }
};

} // namespace st
} // namespace world

#include "tree_search.ipp"
