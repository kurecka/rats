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
    float descent_thd;
    float exploration_constant;
    float risk_exploration_ratio;
    float gamma;
    float gammap;
    environment_handler<S, A>& handler;
    predictor_manager<S, A> predictor;
    bool use_predictor;
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

template<typename SN>
struct select_action_pareto {
    size_t operator()(SN* node, bool explore) const {
        float risk_thd = node->common_data->sample_risk_thd;
        float c = node->common_data->exploration_constant;

        EPC merged_curve;
        EPC* curve_ptr = &node->v.curve;

        std::vector<float> p_ucts;

        if (explore) {
            curve_ptr = &merged_curve;
            // compute uct bonuses
            std::vector<EPC> action_curves;
            std::vector<EPC*> curve_ptrs;
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
                p_ucts.push_back(p_uct);
                action_curves.back().add_and_fix(r_uct, p_uct);
            }
            for (auto& curve : action_curves) {
                curve_ptrs.push_back(&curve);
            }
            merged_curve = convex_hull_merge(curve_ptrs);
        }

        auto [idx, descent_thd] = curve_ptr->select_vertex(risk_thd, node->common_data->risk_exploration_ratio, explore);
        auto& [r, p, supp] = curve_ptr->points[idx];
        auto& [o, thd] = supp.support[0];

        if (explore) {
            descent_thd = std::min(1.f, descent_thd + 2*p_ucts[o]);
        }
        node->common_data->descent_thd = descent_thd;
        return o;
    }
};


/**
 * @brief Update the nodes after a descent
 * 
 * Update sample risk threshold after a descent through an action and a state node.
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
        // float risk_thd = 
        auto [point_idx, risk_thd] = action->q.curve.select_vertex(common_data->descent_thd, common_data->risk_exploration_ratio);
        size_t state_idx = action->child_idx[s];
        // if (debug){
        //     spdlog::trace("Descent: risk_thd={}, point_idx={}, state_idx={}", risk_thd, point_idx, state_idx);
        //     for (auto [s, idx] : action->child_idx) {
        //         spdlog::trace("Child: s={}, idx={}", to_string(s), idx);
        //     }
        // }
        auto& [r, p, supp] = action->q.curve.points[point_idx];
        if (supp.support.size() > state_idx) {
            // if (debug){
            //     for (auto& [o, thd] : supp.support) {
            //         spdlog::trace("Supp: o={}, thd={}", to_string(o), thd);
            //     }
            // }
            size_t support_idx = 0;
            auto& [o, new_thd] = supp.support[0];
            while(o != state_idx){
                support_idx++;
                std::tie(o, new_thd) = supp.support[support_idx];
            }
            
            // if (debug){
            //     spdlog::trace("Descent: p={}, new_thd={}", p, new_thd);
            // }
            if (risk_thd < p) {
                // p = (p - risk_thd) * 0.1f + risk_thd;
                float overflow_ratio = (risk_thd - p) / p;
                new_thd += overflow_ratio * new_thd;
            } else if (risk_thd > p) {
                // p = (p - risk_thd) * 0.1f + risk_thd;
                float overflow_ratio = (risk_thd - p) / (1 - p);
                new_thd += overflow_ratio * (1 - new_thd);
            }
            // } else if (risk_thd > p) {
            //     new_thd = (new_thd - risk_thd) * 0.3f + risk_thd;
            // }
            // if (debug){
            //     spdlog::trace("Descent: update new_thd={}", new_thd);
            // }
            common_data->sample_risk_thd = new_thd;
        }
    }
};

template<typename SN>
void exact_pareto_propagate(SN* leaf) {
    using state_node_t = SN;
    using S = typename SN::S;
    using action_node_t = typename SN::action_node_t;
    using A = typename SN::A;

    state_node_t* current_sn = leaf;

    // ALREADY IN ROLLOUT
    if (!current_sn->common_data->use_predictor) {
        std::get<0>(current_sn->v.curve.points[0]) = current_sn->rollout_reward;
        std::get<1>(current_sn->v.curve.points[0]) = current_sn->rollout_penalty;
    }
    

    // if (!current_sn->is_terminal()) {
    //     for (auto& child : current_sn->children) {
    //         // spdlog::debug("Rollout: {}", child.rollout_penalty);
    //         std::get<0>(child.q.curve.points[0]) = current_sn->rollout_reward;
    //         std::get<1>(child.q.curve.points[0]) = current_sn->rollout_penalty;
    //     }
    // }

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
        std::vector<float> weights;
        weights.reserve(current_an->children.size());
        S s0 = current_an->parent->state;
        A a1 = current_an->action;
        std::string weights_str;
        std::vector<size_t> state_refs;
        auto probs = current_an->common_data->predictor.predict_probs(s0, a1);
        for (auto& [s1, child] : current_an->children) {
            state_curves.push_back(&(child->v.curve));
            weights.push_back(probs[s1]);
            state_refs.push_back(current_an->child_idx[s1]);
        }
        merged_curve = weighted_merge(state_curves, weights, state_refs);
        ++current_an->num_visits;
        merged_curve.num_samples = current_an->num_visits;
        std::pair<float, float> gammas = {current_an->common_data->gamma, current_an->common_data->gammap};
        merged_curve *= gammas;
        current_an->q.curve = merged_curve;

        current_sn = current_an->get_parent();
    }
}


template<typename SN>
void pareto_predictor_rollout(SN* leaf, float thd) {
    using S = typename SN::S;
    using A = typename SN::A;
    predictor_manager<S, A>& predictor = leaf->common_data->predictor;
    S s = leaf->state;
    std::vector<float> thds = {0, thd / 2, thd, thd + (1 - thd) / 2, 1};
    leaf->v.curve.points.clear();
    for (float t : thds) {
        leaf->v.curve.points.emplace_back(predictor.predict_value(t, s), t, outcome_support());
    }
    leaf->v.curve.points = upper_hull(leaf->v.curve.points);
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
    using v_t = pareto_value;
    using q_t = pareto_value;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;
    using action_node_t = action_node<S, A, data_t, v_t, q_t>;
    
    using select_action_t = select_action_pareto<state_node_t>;
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
    float gamma;
    bool use_predictor;

    data_t common_data;

    int graphviz_depth = -1;
    std::string dot_tree;

    std::unique_ptr<state_node_t> root;

    std::vector<std::tuple<S, A, float, float>> episode_history;
public:
    pareto_uct(
        environment_handler<S, A> _handler,
        int _max_depth, float _risk_thd, float _gamma, float _gammap = 1,
        int _num_sim = 100, int _sim_time_limit = 0,
        float _exploration_constant = 5.0, float _risk_exploration_ratio = 1, int _graphviz_depth = -1,
        bool _use_predictor = false
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , sim_time_limit(_sim_time_limit)
    , risk_thd(_risk_thd)
    , gamma(_gamma)
    , use_predictor(_use_predictor)
    , common_data({_risk_thd, _risk_thd, 0, _exploration_constant, _risk_exploration_ratio, _gamma, _gammap, agent<S, A>::handler, {}, _use_predictor})
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

    void simulate(int i) {
        common_data.sample_risk_thd = common_data.risk_thd;
        state_node_t* leaf = select_leaf_f(root.get(), true, max_depth);
        expand_state(leaf);
        if (use_predictor) {
            pareto_predictor_rollout(leaf, common_data.sample_risk_thd);
        } else {
            rollout(leaf);
        }
        pareto_predictor_rollout(leaf, common_data.sample_risk_thd);
        propagate_f(leaf);
        agent<S, A>::handler.end_sim();
    }

    void play() override {
        spdlog::debug("Play: {}", name());
        if (sim_time_limit > 0) {
            auto start = std::chrono::high_resolution_clock::now();
            auto end = start + std::chrono::milliseconds(sim_time_limit);
            int i = 0;
            while (std::chrono::high_resolution_clock::now() < end) {
                simulate(i++);
            }
        } else {
            for (int i = 0; i < num_sim; i++) {
                simulate(i);
            }
        }

        common_data.sample_risk_thd = common_data.risk_thd;
        A a = select_action_f(root.get(), false);
        if (graphviz_depth > 0) {
            dot_tree = to_graphviz_tree(*root.get(), graphviz_depth);
        }

        auto [s, r, p, t] = agent<S, A>::handler.play_action(a);
        episode_history.push_back({root->state, a, r, common_data.sample_risk_thd});

        action_node_t* an = root->get_child(a);
        if (an->children.find(s) == an->children.end()) {
            full_expand_action(an, s, r, p, t);
        }

        std::unique_ptr<state_node_t> new_root = an->get_child_unique_ptr(s);

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
    }

    std::string name() const override {
        return "pareto_uct";
    }

    constexpr bool is_trainable() const override {
        return true;
    }

    void train() {
        spdlog::debug("Train: {}", name());
        // iterate backwards through the episode history
        float cumulative_reward = 0;
        for (auto it = episode_history.rbegin(); it != episode_history.rend(); ++it) {
            cumulative_reward *= common_data.gamma;
            auto [s, a, r, p] = *it;
            cumulative_reward += r;
            common_data.predictor.add_value(cumulative_reward, p, s);
        }
        episode_history.clear();
    }
};

} // namespace ts

} // namespace rats


#include "test.hpp"
