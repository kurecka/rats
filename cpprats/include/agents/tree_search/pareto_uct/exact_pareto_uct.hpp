#pragma once

#include <string>
#include <vector>
#include <array>
#include <cmath>

#include "../tree_search.hpp"
#include "utils.hpp"
#include "pareto_curves.hpp"

namespace rats {
namespace ts {

template <typename S, typename A>
struct pareto_uct_data {
    float risk_thd;
    float sample_risk_thd;
    float exploration_constant;
    float risk_exploration_ratio;
    float gamma;
    float gammap;
    int num_steps;
    environment_handler<S, A>& handler;
    predictor_manager<S, A> predictor;
    mixture mix;
    float lambda;
    float max_disc_penalty = 0;
};

struct pareto_value {
    EPC curve;
    float risk_thd;
};
} // namespace ts

std::string to_string(const ts::pareto_value& v) {
    return to_string(v.curve);
}

} // namespace rats

#include "../string_utils.hpp"

namespace rats {
namespace ts {

template<typename SN, bool use_lambda = false>
struct select_action_pareto {
    size_t operator()(SN* node, bool explore) const {
        float c = node->common_data->exploration_constant;
        float risk_thd = node->common_data->sample_risk_thd;

        // If exploring, a new uct curve must be built
        EPC merged_curve;
        // Pointer to the curve used for selection
        EPC* curve_ptr = &node->v.curve;

        if (explore) {
            // If exploring, the curve is built from the action curves of the children plus the uct bonuses
            curve_ptr = &merged_curve;
            // Compute uct bonuses
            std::vector<EPC> action_curves;
            std::vector<EPC*> curve_ptrs;
            float l_reward = std::numeric_limits<float>::infinity();
            float h_reward = -std::numeric_limits<float>::infinity();
            for (auto& child : node->children) {
                auto [l, h] = child.q.curve.reward_bounds();
                l_reward = std::min(l_reward, l);
                h_reward = std::max(h_reward, h);
            }
            // If the range of rewards is large enough, rescale the exploration constant
            if (h_reward >= l_reward + 1) {
                c *= (h_reward - l_reward);
            }

            // Find least penalty on a pareto curve
            float min_penalty = std::numeric_limits<float>::infinity();
            for (auto& child : node->children) {
                auto [l_penalty, h_penalty] = child.q.curve.penalty_bounds();
                min_penalty = std::min(min_penalty, l_penalty);
            }

            for (auto& child : node->children) {
                action_curves.push_back(child.q.curve);
                // Reward bonus
                float r_uct;
                // Penalty bonus
                float p_uct = 0;
                if (child.q.curve.num_samples == 0) {
                    r_uct = std::numeric_limits<float>::infinity();
                } else {
                    r_uct = c * sqrt(log(node->num_visits + 1) / (child.num_visits + 0.0001));
                    if (use_lambda || min_penalty >= risk_thd) {
                        p_uct = - c * sqrt(log(node->num_visits + 1) / (child.num_visits + 0.0001));
                    }
                }
                // Add the uct bonuses to the action curve
                // If the resulting penalty is negative, set it to zero
                action_curves.back().add_and_fix(r_uct, p_uct);
            }
            // Merge the action curves using convex hull merge
            for (auto& curve : action_curves) {
                curve_ptrs.push_back(&curve);
            }
            merged_curve = convex_hull_merge(curve_ptrs);
        }

        if constexpr (use_lambda) {
            float lambda = node->common_data->lambda;
            node->common_data->mix = curve_ptr->select_vertex_by_lambda<true>(risk_thd, lambda);
        } else {
            // Choose the optimal stochastic mixture of two vertices on the curve based on the risk threshold
            node->common_data->mix = curve_ptr->select_vertex<true>(risk_thd);
        }
        
        return node->common_data->mix.sample();
    }
};


/**
 * @brief Compute the expected immediate penalty of an action
*/
template<typename SN>
float compute_immediate_penalty(SN* sn, typename SN::A a) {
    auto common_data = sn->common_data;

    // Compute immediate penalty (the expected penalty of the action)
    float immediate_penalty = 0;
    auto outcomes = common_data->predictor.predict_probs(sn->state, a);
    auto signals = common_data->predictor.predict_signals(sn->state, a);
    for (auto& [outcome_state, outcome_prob] : outcomes) {
        float observed_penalty = std::get<1>(signals[outcome_state]);
        immediate_penalty += outcome_prob * observed_penalty;
    }

    return immediate_penalty;
}


/**
 * @brief Update the nodes after a descent
 * 
 * @tparam SN State node type
 * @tparam debug Debug flag
 * 
 * @param s0 Parent state node
 * @param a Action taken
 * @param action Action node
 * @param s New state
 * @param new_state New state node
 * 
 * Updates the sample risk threshold based on the selected action and the new state.
 */
template<typename SN, bool debug>
struct descend_callback {
    using state_node_t = SN;
    using S = typename SN::S;
    using action_node_t = typename SN::action_node_t;
    using A = typename SN::A;
    using DATA = typename SN::DATA;
    void operator()(state_node_t* s0, A a, action_node_t* action, S s, state_node_t* new_state) const {
        DATA* common_data = action->common_data;

        // Compute the intermediate action penalty budget
        float action_thd = common_data->mix.update_thd(common_data->sample_risk_thd);
        if constexpr (debug) {
            spdlog::debug("Action thd: {}", action_thd);
        }
        // Find a mixture of the points on the action q curve
        auto mix = action->q.curve.select_vertex(action_thd);
        size_t state_idx = action->child_idx[s];
        // Retrieve the curve point data
        auto& [r1, point_penalty0, supp0] = action->q.curve.points[mix.vals[0]];
        auto& [r2, point_penalty1, supp1] = action->q.curve.points[mix.vals[1]];

        // If the state index is valid, update the sample risk threshold
        if (supp0.num_outcomes() > state_idx) {
            // The points on the action curve contain the immediate penalty and discount the future by gammap
            float state_penalty0 = (supp0.penalty_at_outcome(state_idx) - new_state->observed_penalty) / common_data->gammap;
            float state_penalty1 = (supp1.penalty_at_outcome(state_idx) - new_state->observed_penalty) / common_data->gammap;

            float immediate_penalty = compute_immediate_penalty(s0, a);

            if (mix.vals[0] != mix.vals[1]) {
                // If a proper mixing is happening, compute the expected penalty
                common_data->sample_risk_thd = mix.expectation(state_penalty0, state_penalty1);
            } else if (point_penalty0 > action_thd) {
                // If the minimum playable penalty is greater than the action threshold, decrease the threshold
                // so that all missing budget is removed from the played branch
                float outcome_prob = common_data->predictor.predict_probs(s0->state, a)[s];
                common_data->sample_risk_thd = state_penalty0 - (point_penalty0 - action_thd) / (common_data->gammap * outcome_prob);
                common_data->sample_risk_thd = std::max(common_data->sample_risk_thd, 0.0f);
            } else {
                // If the minimum playable penalty is less than the action threshold, distribute the remaining budget to all branches
                // so that the maximum penalty is never exceeded
                float overflow_ratio = 0;
                if (point_penalty1 < action_thd) {
                    overflow_ratio = (action_thd - point_penalty1) / (common_data->gammap * common_data->max_disc_penalty - point_penalty1 + immediate_penalty);
                }
                common_data->sample_risk_thd = state_penalty1 + overflow_ratio * (common_data->max_disc_penalty - state_penalty1);
            }
        }
    }
};

/**
 * @brief Build the pareto curve for a leaf node
 * 
 * @tparam SN State node type
 * 
 * @param leaf Leaf state node to build the curve for
 * 
 * Builds the pareto curve from the rollout reward and penalty and the observed reward and penalty.
 * 
 * If the rollout penalty is zero, the curve is initialized with a single point (rollout_reward, rollout_penalty).
 * If the rollout policy is non-zero, the curve is initialized with a two points [(0,0), (rollout_reward, rollout_penalty)].
*/
template<typename SN>
void build_leaf_curve(SN* leaf) {
    // If the curve is already built, return
    if (leaf->v.curve.num_samples > 0) {
        return;
    }
    if (leaf->rollout_penalty <= 0) {
        // If the rollout penalty is zero, initialize the curve with a single point
        leaf->v.curve.points[0] = {
            leaf->rollout_reward,
            leaf->rollout_penalty,
            {}
        };
    } else if (leaf->rollout_reward > 0) {
        // If the rollout penalty is non-zero, initialize the curve with two points
        leaf->v.curve.points.push_back({
            leaf->rollout_reward,
            leaf->rollout_penalty,
            {}
        });
    }

    // Add the incoming observed reward and penalty
    leaf->v.curve *= {leaf->common_data->gamma, leaf->common_data->gammap};
    leaf->v.curve += {leaf->observed_reward, leaf->observed_penalty};
    leaf->v.curve.num_samples = 1;
}


/**
 * @brief Propagate the pareto curve estimates up the tree
 * 
 * @tparam SN State node type
 * 
 * @param leaf Leaf state node to start the propagation from
 * 
 * Propagates the pareto curve estimates up the tree by merging the curves of the children
 * and updating the value of the parent node.
 * 
 * The propagation is done in two steps:
 * 1. Merge the state curves of the children of an action node
 * 2. Merge the action curves of the children of a state node
 */
template<typename SN>
void exact_pareto_propagate(SN* leaf) {
    using state_node_t = SN;
    using S = typename SN::S;
    using action_node_t = typename SN::action_node_t;
    using A = typename SN::A;

    state_node_t* current_sn = leaf;

    // Build the leaf curve
    build_leaf_curve(current_sn);

    while (!current_sn->is_root()) {
        // ----------------- merge state curves -----------------
        action_node_t* current_an = current_sn->get_parent();
        std::vector<EPC*> state_curves;
        std::vector<float> weights;
        weights.reserve(current_an->children.size());
        S s0 = current_an->parent->state;
        A a1 = current_an->action;
        std::vector<size_t> state_refs;
        // probs[s1] = p(s1 | s0, a1)
        auto probs = current_an->common_data->predictor.predict_probs(s0, a1);
        for (auto& [s1, child] : current_an->children) {
            // Build the child leaf curve if needed
            build_leaf_curve(child.get());
            // Add the child state curve to the list of curves to merge
            state_curves.push_back(&(child->v.curve));
            weights.push_back(probs[s1]);
            state_refs.push_back(current_an->child_idx[s1]);
        }
        // Use weighted merge to merge the state curves
        // Area under the merged curve is the weighted Minkowski sum of the areas under the input curves
        EPC merged_curve = weighted_merge(state_curves, weights, state_refs);
        ++current_an->num_visits;
        merged_curve.num_samples = current_an->num_visits;
        current_an->q.curve = merged_curve;

        // ----------------- merge action curves -----------------
        current_sn = current_an->parent;
        std::vector<EPC*> action_curves;
        for (auto& child : current_sn->children) {
            // Add the child action curve to the list of curves to merge
            action_curves.push_back(&child.q.curve);
        }
        // Convex hull merge the action curves
        // Area under the merged curve is the convex closure of the union of areas under the input curves
        merged_curve = convex_hull_merge(action_curves);
        ++current_sn->num_visits;
        merged_curve.num_samples = current_sn->num_visits;
        current_sn->v.curve = merged_curve;

        // State curves contains the incoming observed reward and penalty (makes the implementation simpler)
        current_sn->v.curve *= {current_sn->common_data->gamma, current_sn->common_data->gammap};
        current_sn->v.curve += {current_sn->observed_reward, current_sn->observed_penalty};
    }
}


// /*********************************************************************
//  * @brief Pareto uct agent
//  * 
//  * @tparam S State type
//  * @tparam A Action type
//  *********************************************************************/

template <typename S, typename A, bool use_lambda = false>
class pareto_uct : public agent<S, A> {
    using data_t = pareto_uct_data<S, A>;
    using v_t = pareto_value;
    using q_t = pareto_value;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;
    using action_node_t = action_node<S, A, data_t, v_t, q_t>;
    
    using select_action_t = select_action_pareto<state_node_t, use_lambda>;
    constexpr static auto select_action_f = select_action_pareto<state_node_t>();
    using descend_callback_t = descend_callback<state_node_t, false>;
    constexpr static auto descend_callback_f = descend_callback<state_node_t, true>();
    constexpr static auto select_leaf_f = select_leaf<state_node_t, select_action_t, descend_callback_t>;
    constexpr static auto propagate_f = exact_pareto_propagate<state_node_t>;
private:
    int max_depth;
    int num_sim;
    int sim_time_limit;
    float risk_thd;
    bool use_rollout;
    float lambda;
    float lambda_max = 100;
    double grad_buffer = 0;

    data_t common_data;

    int graphviz_depth = -1;
    std::string dot_tree;

    std::unique_ptr<state_node_t> root;
public:
    pareto_uct(
        environment_handler<S, A> _handler,
        int _max_depth, float _risk_thd, float _gamma, float _gammap = 1, float _max_disc_penalty = 1,
        int _num_sim = 100, int _sim_time_limit = 0,
        float _exploration_constant = 5.0, float _risk_exploration_ratio = 1, 
        bool _rollout = true,
        int _graphviz_depth = -1,
        float _lambda = -1
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , sim_time_limit(_sim_time_limit)
    , risk_thd(_risk_thd)
    , use_rollout(_rollout)
    , lambda(_lambda)
    , common_data({_risk_thd, _risk_thd, _exploration_constant, _risk_exploration_ratio, _gamma, _gammap, 0, agent<S, A>::handler, {}, {}, lambda, _max_disc_penalty})
    , graphviz_depth(_graphviz_depth)
    , root(std::make_unique<state_node_t>())
    {
        reset();
    }

    std::string get_graphviz() const {
        return dot_tree;
    }

    std::string get_state_curve(S s) {
        std::vector<float> thds = {0, 0.25, 0.5, 0.75, 1};
        std::string curve_str = "";
        for (float t : thds) {
            curve_str += fmt::format("{} ", common_data.predictor.predict_value(t, s));
        }
        return curve_str;
    }

    /**
     * @brief Perform i-th simulation of the MCTS algorithm
     * 
     * @param i Simulation number
     * 
     * Consists of the following steps:
     * 1. Selection: Select a leaf node using `select_action_pareto` and `descend_callback`
     * 2. Expansion: Expand the selected leaf node
     * 3. Rollout: Perform a rollout from the leaf node
     * 4. Backpropagation: Propagate the values up the tree using `exact_pareto_propagate`
     * (5. End simulation: Call the end_sim callback)
     * 6. Perform a gradient step on the lambda parameter (if enabled)
     */
    void simulate(int i) {
        common_data.sample_risk_thd = common_data.risk_thd;
        state_node_t* leaf = select_leaf_f(root.get(), true, max_depth);
        expand_state(leaf);
        if (use_rollout) {
            rollout(leaf, true);
        }
        propagate_f(leaf);
        agent<S, A>::handler.end_sim();
        if constexpr (use_lambda) {
            select_action_t()(root.get(), false);
            mixture& mix = root->common_data->mix;

            double gradient = mix.last_penalty() - common_data.risk_thd;
            grad_buffer += (gradient - grad_buffer) * 0.3;

            double eps = 1;
            eps = 1.0 / sqrt(1.0 + i);
            common_data.lambda += grad_buffer * eps;
            common_data.lambda = std::clamp(common_data.lambda, 0.f, lambda_max);
        }
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
   	    spdlog::debug("Number of simulations: {}", i);
        } else {
            for (int i = 0; i < num_sim; i++) {
                simulate(i);
            }
        }

        // Sample risk action is modified during the simulations -> reset it
        common_data.sample_risk_thd = common_data.risk_thd;

        // Select the best action
        size_t a_idx = select_action_t()(root.get(), false);
        A a = root->actions[a_idx];
        if (graphviz_depth > 0) {
            dot_tree = to_graphviz_tree(*root.get(), graphviz_depth);
        }

        // Play the selected action
        auto [s, r, p, t] = agent<S, A>::handler.play_action(a);
        ++common_data.num_steps;

        spdlog::debug("Play action: {}", to_string(a));
        spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

        // If the state is not in the tree, update the predictor and expand the action node
        action_node_t* an = root->get_child(a_idx);
        if (an->children.find(s) == an->children.end()) {
            update_predictor(root.get(), a, s, r, p, t);
            full_expand_action(an);
        }

        std::unique_ptr<state_node_t> new_root = an->get_child_unique_ptr(s);

        // Descend to the new root
        // Update the risk threshold by calling the descend callback
        float old_risk_thd = common_data.risk_thd;
        descend_callback_f(root.get(), a, an, s, new_root.get());
        common_data.risk_thd = common_data.sample_risk_thd;
        spdlog::debug("Risk thd: {} -> {}", old_risk_thd, common_data.risk_thd);

        root = std::move(new_root);
        root->get_parent() = nullptr;
    }

    void reset() override {
        spdlog::debug("Reset: {}", name());
        agent<S, A>::reset();
        common_data.risk_thd = common_data.sample_risk_thd = risk_thd;
        root = std::make_unique<state_node_t>();
        root->common_data = &common_data;
        root->state = agent<S, A>::handler.get_current_state();
        common_data.handler.gamma = common_data.gamma;
        common_data.handler.gammap = common_data.gammap;
        common_data.num_steps = 0;
        common_data.lambda = lambda;
    }

    std::string name() const override {
        if constexpr (use_lambda) {
            return "lambda_pareto_uct";
        } else {
            return "pareto_uct";    
        }
    }
};

} // namespace ts

} // namespace rats
