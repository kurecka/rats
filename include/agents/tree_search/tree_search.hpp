#pragma once

#include "tree_logger.hpp"

#include <vector>
#include <map>

namespace gym {
namespace ts {

template<typename S, typename A, typename SN, typename AN>
concept CompatibleNodes = requires (S s, A a, SN sn, AN an, float f, bool b, size_t n)
{
    {sn.select_action(b)} -> std::same_as<A>;
    {sn.propagate(&an, f)};
    {sn.get_child(a)} -> std::same_as<AN*>;
    {sn.get_parent()} -> std::same_as<AN*&>;
    {sn.get_num_visits()} -> std::same_as<size_t>;
    {sn.is_terminal()} -> std::same_as<bool>;
    {sn.to_string()} -> std::same_as<std::string>;
    {sn.descend_update(a, s, b)};

    {an.add_outcome(s, f, f, b)};
    {an.propagate(&sn, f)};
    {an.get_child(s)} -> std::same_as<SN*>;
    {an.get_child_unique_ptr(s)} -> std::same_as<std::unique_ptr<SN>&&>;
    {an.get_parent()} -> std::same_as<SN*&>;
    {an.get_num_visits()} -> std::same_as<size_t>;
    {an.to_string()} -> std::same_as<std::string>;
};

template<typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
bool is_root(SN* sn) {
    return sn->get_parent() == nullptr;
}

template<typename S, typename A, typename SN, typename AN>
requires CompatibleNodes<S, A, SN, AN>
bool is_leaf(SN* sn) {
    return sn->children.empty();
}



/*********************************************************************
 * @brief Search tree
 * 
 * @tparam S type of states
 * @tparam SN type of state nodes
 * @tparam AN type of action nodes
 */
template <typename S, typename A, typename SN, typename AN, typename DATA>
requires CompatibleNodes<S, A, SN, AN>
class tree_search {
public:
    using state_node_t = SN;
    using action_node_t = AN;
    using env_t = environment<S, A>;
    using handler_t = environment_handler<S, A>;

private:
    int max_depth;
    float gamma; // discount factor

    std::unique_ptr<state_node_t> root;

    DATA* common_data;

public:
    /**
     * @brief Get the root node of the search tree
     * 
     * @return state_node_t*
     */
    state_node_t* get_root() { return root.get(); }

    /**
     * @brief Select a leaf node to expand.
     * 
     * @return state_node_t*
     */
    state_node_t* select();

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
    void descent(A a, S s);

public:
    /**
     * @brief Construct a new search tree object
     * 
     * @param _max_depth Maximum depth of the search tree
     * @param _gamma Discount factor
     */
    tree_search( int _max_depth, float _gamma, DATA* _common_data)
    : max_depth(_max_depth), gamma(_gamma), common_data(_common_data) {
        reset();
    }

    void reset() {
        root = std::make_unique<state_node_t>();
        root->get_parent() = nullptr;
    }
};

} // namespace st
} // namespace gym

#include "tree_search.ipp"
