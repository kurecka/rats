#pragma once

#include <string>
#include <vector>
#include <array>
#include <cmath>

#include "pareto_curves.hpp"
#include "utils.hpp"

namespace rats {
namespace ts {

template <typename S, typename A>
class prob_predictor {
    std::map<std::tuple<S, A, S>, size_t> counts;
    std::map<std::pair<S, A>, size_t> total_counts;
public:
    void add(S s0, A a1, S s1) {
        counts[{s0, a1, s1}] += 1;
        total_counts[{s0, a1}] += 1;
    }

    float predict(S s0, A a1, S s1) {
        return static_cast<float>(counts[{s0, a1, s1}]) / total_counts[{s0, a1}];
    }
};

template <typename S, typename A>
class exact_prob_predictor {
    std::map<std::tuple<S, A, S>, float> probs;
public:
    void add(S s0, A a1, S s1, float p) {
        probs[{s0, a1, s1}] = p;
    }

    float predict(S s0, A a1, S s1) {
        return probs[{s0, a1, s1}];
    }
};

template <typename S, typename A>
struct pareto_uct_data {
    float risk_thd;
    float sample_risk_thd;
    size_t descent_point;
    float exploration_constant;
    environment_handler<S, A>& handler;
    exact_prob_predictor<S, A> predictor;
    float gamma;
};


template <typename pareto_curve>
struct pareto_value {
    pareto_curve curve;
    float risk_thd;
};


template <typename pareto_curve>
std::string to_string(pareto_value<pareto_curve> v) {
    return to_string(v.curve);
}


template<typename S, typename A, typename DATA, typename pareto_curve>
A select_action_pareto(state_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>* node, bool explore) {
    float risk_thd = node->common_data->sample_risk_thd;
    float c = node->common_data->exploration_constant;

    pareto_curve merged_curve;
    pareto_curve* curve_ptr = &node->v.curve;

    if (explore) {
        curve_ptr = &merged_curve;
        // compute uct bonuses
        std::vector<pareto_curve> action_curves;
        std::vector<pareto_curve*> curve_ptrs;
        for (auto& child : node->children) {
            action_curves.push_back(child.q.curve);
            float r_uct;
            float p_uct;
            if (child.q.curve.num_samples == 0) {
                r_uct = std::numeric_limits<float>::infinity();
                p_uct = 0;
            } else {
                r_uct = c * powf(static_cast<float>(node->num_visits), 0.8f) / child.q.curve.num_samples;
                p_uct = -c * powf(static_cast<float>(node->num_visits), 0.6f) / child.q.curve.num_samples;
            }
            action_curves.back().add_and_fix(r_uct, p_uct);
        }
        for (auto& curve : action_curves) {
            curve_ptrs.push_back(&curve);
        }
        merged_curve = convex_hull_merge(curve_ptrs);
    }

    size_t idx;
    // TODO: bin search
    for (idx = 0; idx < curve_ptr->points.size()-1; ++idx) {
        if (std::get<1>(curve_ptr->points[idx + 1]) > risk_thd) {
            break;
        }
    }


    size_t opt_vertex_idx;

    if (idx == curve_ptr->points.size() - 1) {
        opt_vertex_idx = curve_ptr->points.size() - 1;
    } else {
        float p1 = std::get<1>(curve_ptr->points[idx]);
        float p2 = std::get<1>(curve_ptr->points[idx + 1]);

        float prob2 = (risk_thd - p1) / (p2 - p1);

        if (p1 > risk_thd) {
            opt_vertex_idx = idx;
        } if (rng::unif_float() < prob2) {
            opt_vertex_idx = idx + 1;
        } else {
            opt_vertex_idx = idx;
        }
    }
    auto& [r, p, supp] = curve_ptr->points[opt_vertex_idx];
    auto [o, vtx] = supp.support[0];

    node->common_data->descent_point = vtx;
    return o;
}


/**
 * @brief Update the nodes after a descent
 * 
 * Update sample risk threshold after a descent through an action and a state node.
 */
template<typename S, typename A, typename DATA, typename pareto_curve>
void descend_callback(
    state_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>* s0,
    A a, action_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>* action,
    S s, state_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>* new_state
) {
    DATA* common_data = action->common_data;

    // common_data->predictor.add(s0->state, a, s);
    if (new_state->is_leaf()) {
        auto probs = common_data->handler.outcome_probabilities(s0->state, a);
        common_data->predictor.add(s0->state, a, s, probs[s]);
    }

    size_t point_idx = common_data->descent_point;
    size_t state_idx = action->child_idx[s];
    auto& [r, p, supp] = action->q.curve.points[point_idx];
    if (supp.support.size() > state_idx) {
        auto& [o, vtx] = supp.support[state_idx];
        common_data->sample_risk_thd = std::get<1>(new_state->v.curve.points[vtx]);
    }
}

template<typename S, typename A, typename DATA, typename V, typename Q>
void exact_pareto_propagate(state_node<S, A, DATA, V, Q>* leaf, float gamma) {
    using state_node_t = state_node<S, A, DATA, V, Q>;
    using action_node_t = action_node<S, A, DATA, V, Q>;

    action_node_t* prev_an = nullptr;
    state_node_t* current_sn = leaf;

    // TODO: rolout
    float disc_r = leaf->rollout_reward;
    float disc_p = leaf->rollout_penalty;

    for (auto& child : current_sn->children) {
        std::get<0>(child.q.curve.points[0]) = disc_r;
        std::get<1>(child.q.curve.points[0]) = disc_p;
    }

    while (true) {
        std::vector<EPC*> action_curves;
        for (auto& child : current_sn->children) {
            action_curves.push_back(&child.q.curve);
        }
        EPC merged_curve = convex_hull_merge(action_curves);
        ++current_sn->num_visits;
        merged_curve.num_samples = current_sn->num_visits;
        merged_curve += {current_sn->observed_reward, current_sn->observed_penalty};
        current_sn->v.curve = merged_curve;

        if (current_sn->is_root()) {
            break;
        }

        action_node_t* current_an = current_sn->get_parent();
        std::vector<EPC*> state_curves;
        for (auto& [s, child] : current_an->children) {
            state_curves.push_back(&(child->v.curve));
        }
        std::vector<float> weights;
        weights.reserve(current_an->children.size());
        auto& predictor = current_an->common_data->predictor;
        S s0 = current_an->parent->state;
        A a1 = current_an->action;
        std::string weights_str;
        for (auto& [s1, child] : current_an->children) {
            weights.push_back(predictor.predict(s0, a1, s1));
            weights_str += "s(" + std::to_string(s1) + ")=" + std::to_string(weights.back()) + ", ";
        }
        
        spdlog::debug("S = {}, A = {}", s0, a1);
        spdlog::debug("Weights: {}", weights_str);
        merged_curve = weighted_merge(state_curves, weights);
        ++current_an->num_visits;
        merged_curve.num_samples = current_an->num_visits;
        merged_curve *= gamma;
        current_an->q.curve = merged_curve;

        current_sn = current_an->get_parent();
        prev_an = current_an;
    }
}


// /*********************************************************************
//  * @brief Pareto uct agent
//  * 
//  * @tparam S State type
//  * @tparam A Action type
//  *********************************************************************/

template <typename S, typename A>
class pareto_uct : public agent<S, A> {
    using data_t = pareto_uct_data<S, A>;
    using v_t = pareto_value<EPC>;
    using q_t = pareto_value<EPC>;
    using pareto_curve = EPC;
    using uct_state_t = state_node<S, A, data_t, v_t, q_t>;
    using uct_action_t = action_node<S, A, data_t, v_t, q_t>;
    
    constexpr static auto select_action_f = select_action_pareto<S, A, data_t, pareto_curve>;
    constexpr static auto descend_callback_f = descend_callback<S, A, data_t, pareto_curve>;
    constexpr static auto select_leaf_f = select_leaf<S, A, data_t, v_t, q_t, select_action_f, descend_callback_f>;
    constexpr static auto propagate_f = exact_pareto_propagate<S, A, data_t, v_t, q_t>;
private:
    int max_depth;
    int num_sim;
    float risk_thd;
    float gamma;

    data_t common_data;

    int graphviz_depth = -1;
    std::string dot_tree;

    std::unique_ptr<uct_state_t> root;
public:
    pareto_uct(
        environment_handler<S, A> _handler,
        int _max_depth, int _num_sim, float _risk_thd, float _gamma,
        float _exploration_constant = 5.0, int _graphviz_depth = -1
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , risk_thd(_risk_thd)
    , gamma(_gamma)
    , common_data({_risk_thd, _risk_thd, 0, _exploration_constant, agent<S, A>::handler, {}, gamma})
    , graphviz_depth(_graphviz_depth)
    , root(std::make_unique<uct_state_t>())
    {
        reset();
    }

    std::string get_graphviz() const {
        return dot_tree;
    }

    void play() override {
        spdlog::debug("Play: {}", name());

        for (int i = 0; i < num_sim; i++) {
            spdlog::trace("Simulation {}", i);
            common_data.sample_risk_thd = common_data.risk_thd;
            spdlog::trace("Select");
            uct_state_t* leaf = select_leaf_f(root.get(), true, max_depth);
            spdlog::trace("Expand");
            expand_state(leaf);
            spdlog::trace("Rollout");
            constant_rollout<S, A, data_t, v_t, q_t, 1, 100>(leaf);
            spdlog::trace("Propagate");
            propagate_f(leaf, gamma);
            spdlog::trace("Reset");
            agent<S, A>::handler.sim_reset();
        }

        common_data.sample_risk_thd = common_data.risk_thd;
        A a = select_action_pareto<S, A, data_t, pareto_curve>(root.get(), false);

        if (graphviz_depth > 0) {
            spdlog::debug("Graphviz checkpoint: {}", graphviz_depth);
            dot_tree = to_graphviz_tree(*root.get(), graphviz_depth);
        }

        auto [s, r, p, t] = agent<S, A>::handler.play_action(a);
        spdlog::debug("Play action: {}", a);
        spdlog::debug(" Result: s={}, r={}, p={}", s, r, p);

        uct_action_t* an = root->get_child(a);
        if (an->children.find(s) == an->children.end()) {
            expand_action(an, s, r, p, t);
        }

        std::unique_ptr<uct_state_t> new_root = an->get_child_unique_ptr(s);

        descend_callback_f(root.get(), a, an, s, new_root.get());

        root = std::move(new_root);
        root->get_parent() = nullptr;
    }

    void reset() override {
        spdlog::debug("Reset: {}", name());
        agent<S, A>::reset();
        common_data.risk_thd = common_data.sample_risk_thd = risk_thd;
        root = std::make_unique<uct_state_t>();
        root->common_data = &common_data;
        root->state = agent<S, A>::handler.get_current_state();
    }

    std::string name() const override {
        return "pareto_uct";
    }
};

} // namespace ts
} // namespace rats


#include "test.hpp"
