#pragma once

#include "string_utils.hpp"
#include "rand.hpp"
#include "lp.hpp"
#include <string>
#include <vector>
#include <map>

namespace rats {
namespace ts {

template <typename S, typename A>
struct ramcp_data {
    float risk_thd;
    float exploration_constant;
    float gamma;
    float gammap;
    environment_handler<S, A>& handler;
    predictor_manager<S, A> predictor;
};


/*********************************************************************
 * @brief Action selection for RAMCP based on UCT
 * 
 * For each action a compute the UCT value according to the formula:
 *  UCT(a) = normQ(a) + c * sqrt(log(N + 1) / (n(a) + 0.0001))
 * where:
 * normQ(a) = (Q(a) - min(Q)) / (max(Q) - min(Q))
 * N is the total number of visits of the parent node
 * n(a) is the number of visits of action a
 * c is the exploration constant
 * 
 * Returns the index of the action with the highest UCT value
 *********************************************************************/

template<typename SN>
struct select_action_uct {
    size_t operator()(SN* node, bool /*explore*/) const {

        float c = node->common_data->exploration_constant;

        auto& children = node->children;
        
        // Use max_v, min_v to normalize the Q values
        float max_v = std::max_element(children.begin(), children.end(),
                        [](auto& l, auto& r){ return l.q.first < r.q.first; })->q.first;
        float min_v = std::min_element(children.begin(), children.end(),
                        [](auto& l, auto& r){ return l.q.first < r.q.first; })->q.first;
        
        // If max_v == min_v, then the normalization does not matter since all values are the same.
        // In this case, we set max_v to min_v + 1 to avoid division by zero.
        if (max_v <= min_v) {
            max_v = min_v + 1;
        }

        size_t idxa = 0;
        float max_uct = 0, uct_value = 0;
        for (size_t i = 0; i < children.size(); ++i) {
            uct_value = ((children[i].q.first - min_v) / (max_v - min_v)) +
                c * static_cast<float>(std::sqrt(std::log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            )  + rng::unif_float(0.0001);  // Add a small random value to break ties

            if (uct_value > max_uct) {
                max_uct = uct_value;
                idxa = i;
            }
        }

        return idxa;
    }
};

/*********************************************************************
 * @brief exact ramcp uct agent
 * 
 * @tparam S State type
 * @tparam A Action type
 *********************************************************************/

template <typename S, typename A>
class ramcp : public agent<S, A> {
    using data_t = ramcp_data<S, A>;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;
    using action_node_t = action_node<S, A, data_t, v_t, q_t>;

    using select_action_t = select_action_uct<state_node_t>;
    using descend_callback_t = void_fn<state_node_t*, A, action_node_t*, S, state_node_t*>;
    constexpr auto static select_action_f = select_action_t();
    constexpr auto static select_leaf_f = select_leaf<state_node_t, select_action_t, descend_callback_t>;
    constexpr auto static propagate_f = propagate<state_node_t, uct_prop_v_value<state_node_t>, uct_prop_q_value<action_node_t>>;

private:
    int max_depth;
    int num_sim;
    int sim_time_limit;
    float risk_thd;
    bool use_rollout;

    data_t common_data;

    int graphviz_depth = -1;
    std::string dot_tree;

    std::unique_ptr<state_node_t> root;
    lp::tree_lp_solver<state_node_t> solver;
public:
    ramcp(
        environment_handler<S, A> _handler,
        int _max_depth, float _risk_thd, float _gamma,
        float _gammap = 1, int _num_sim = 100, int _sim_time_limit = 0,
        float _exploration_constant = 5.0,
        bool _rollout = true,
        int _graphviz_depth = -1
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , sim_time_limit(_sim_time_limit)
    , risk_thd(_risk_thd)
    , use_rollout(_rollout)
    , common_data({_risk_thd, _exploration_constant, _gamma, _gammap, agent<S, A>::handler, {}})
    , graphviz_depth(_graphviz_depth)
    , root(std::make_unique<state_node_t>())
    {
        reset();
    }

    std::string get_graphviz() const {
        return dot_tree;
    }

    /**
     * @brief Perform i-th simulation of the MCTS algorithm
     * 
     * Consists of the following steps:
     * 1. Selection: Select a leaf node using `select_action_uct` and no descent callback
     * 2. Expansion: Expand the leaf node using `expand_state`
     * 3. Rollout: Perform a rollout from the leaf node using `rollout`
     * 4. Backpropagation: Backpropagate the results of the rollout using `uct_prop_v_value` and `uct_prop_q_value`
     * 
     * (5. End simulation: Call the end_sim callback)
     */
    void simulate(int i) {
        state_node_t* leaf = select_leaf_f(root.get(), true, max_depth);
        expand_state(leaf);
        if (use_rollout) {
            rollout(leaf);
        }
        propagate_f(leaf);
        agent<S, A>::handler.end_sim();
    }

    void play() override {
        spdlog::debug("Play: {}", name());

        // Perform simulations: Either based on number of simulations or time limit
        if (sim_time_limit > 0) {
            auto start = std::chrono::high_resolution_clock::now();
            auto end = start + std::chrono::milliseconds(sim_time_limit);
            int i = 0;
            while (std::chrono::high_resolution_clock::now() < end) {
                simulate(i++);
            }
        } else {
            for (int i = 0; i < num_sim; ++i) {
                simulate(i);
            }
        }

        // Run LP solver to get the best safe action according to the sampled tree.
        // The choice of the action is stochastic to balance penalty and reward. The solver remembers
        // the alternative action and other relevant information to update the risk threshold.
        size_t a_idx = solver.get_action(root.get(), risk_thd);
	    A a = root->actions[a_idx];

        // Plot the tree
        if (graphviz_depth > 0) {
            dot_tree = to_graphviz_tree(*root.get(), graphviz_depth);
        }

        // Play the selected action
        auto [s, r, p, t] = common_data.handler.play_action(a);
        spdlog::debug("Play action: {}", to_string(a));
        spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);


        action_node_t* an = root->get_child(a_idx);
        // If the state is not in the tree, add it
        if (an->children.find(s) == an->children.end()) {
            update_predictor(root.get(), a, s, r, p, t);
            full_expand_action(an);
        }

        // Update the penalty threshold
        risk_thd = solver.update_threshold(risk_thd, a, s, p);

        // Update the root node
        std::unique_ptr<state_node_t> new_root = an->get_child_unique_ptr(s);
        root = std::move(new_root);
        root->get_parent() = nullptr;
    }


    /**
     * Reset:
     * - Reset the agent
     * - Reset the risk threshold
     * - Create a new root node
     * - Set the common data for the root node
     * - Set a state for the root node
     * - Set the gamma and gammap for the handler
     */
    void reset() override {
        spdlog::debug("Reset: {}", name());
        agent<S, A>::reset();
        risk_thd = common_data.risk_thd;
        root = std::make_unique<state_node_t>();
        root->common_data = &common_data;
        root->state = common_data.handler.get_current_state();
        common_data.handler.gamma = common_data.gamma;
        common_data.handler.gammap = common_data.gammap;
    }

    std::string name() const override {
        return "ramcp";
    }
};

}
}
