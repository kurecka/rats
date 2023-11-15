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
    bool use_predictor;
    mixture mix;
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

        // if (explore) {
        //     float dev = std::min(0.1, risk_thd * 0.3);

        //     int tree_depth = node->node_depth() - node->common_data->num_steps;
        //     dev = exp(-tree_depth) * dev;
        //     risk_thd = std::clamp(rng::normal(risk_thd, dev), 0.f, 1.f);
        // }

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
                    p_uct = 0;
                    // p_uct = -c * powf(static_cast<float>(node->num_visits), 0.6f) / child.q.curve.num_samples;
                }
                p_ucts.push_back(p_uct);
                action_curves.back().add_and_fix(r_uct, p_uct);
            }
            for (auto& curve : action_curves) {
                curve_ptrs.push_back(&curve);
            }
            merged_curve = convex_hull_merge(curve_ptrs);
        }

        node->common_data->mix = curve_ptr->select_vertex<true>(risk_thd);
        return node->common_data->mix.sample();
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

        float action_thd = common_data->mix.update_thd(common_data->sample_risk_thd);
        auto mix = action->q.curve.select_vertex(action_thd);
        size_t state_idx = action->child_idx[s];
        auto& [r1, point_penalty1, supp1] = action->q.curve.points[mix.vals[0]];
        auto& [r2, point_penalty2, supp2] = action->q.curve.points[mix.vals[1]];

        if (supp1.num_outcomes() > state_idx) {
            float state_penalty1 = supp1.penalty_at_outcome(state_idx);
            float state_penalty2 = supp2.penalty_at_outcome(state_idx);

            if (mix.vals[0] != mix.vals[1]) {
                common_data->sample_risk_thd = mix.expectation(state_penalty1, state_penalty2);
            } else if (mix.penalties[0] > action_thd) {
                float overflow_ratio = (action_thd - point_penalty1) / point_penalty1;
                common_data->sample_risk_thd = state_penalty1 + overflow_ratio * state_penalty1;
            } else {
                float overflow_ratio = (action_thd - point_penalty1) / (1 - point_penalty1);
                common_data->sample_risk_thd = state_penalty1 + overflow_ratio * (1 - state_penalty1);
            }
        }
    }
};

template<typename SN>
void build_leaf_curve(SN* leaf) {
    if (leaf->v.curve.num_samples > 0 || leaf->common_data->use_predictor) {
        return;
    }
    if (leaf->rollout_penalty <= 0) {
        leaf->v.curve.points[0] = {
            leaf->rollout_reward,
            leaf->rollout_penalty,
            {}
        };
    } else if (leaf->rollout_reward > 0) {
        leaf->v.curve.points.push_back({
            leaf->rollout_reward,
            leaf->rollout_penalty,
            {}
        });
    }

    leaf->v.curve *= {leaf->common_data->gamma, leaf->common_data->gammap};
    leaf->v.curve += {leaf->observed_reward, leaf->observed_penalty};
    leaf->v.curve.num_samples = 1;
}

template<typename SN>
void exact_pareto_propagate(SN* leaf) {
    using state_node_t = SN;
    using S = typename SN::S;
    using action_node_t = typename SN::action_node_t;
    using A = typename SN::A;

    state_node_t* current_sn = leaf;

    build_leaf_curve(current_sn);

    while (!current_sn->is_root()) {
        // merge state curves -----------------
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
            build_leaf_curve(child.get());
            state_curves.push_back(&(child->v.curve));
            weights.push_back(probs[s1]);
            state_refs.push_back(current_an->child_idx[s1]);
        }
        EPC merged_curve = weighted_merge(state_curves, weights, state_refs);
        ++current_an->num_visits;
        merged_curve.num_samples = current_an->num_visits;
        current_an->q.curve = merged_curve;

        // merge action curves -----------------
        current_sn = current_an->parent;
        std::vector<EPC*> action_curves;
        for (auto& child : current_sn->children) {
            action_curves.push_back(&child.q.curve);
        }
        merged_curve = convex_hull_merge(action_curves);
        ++current_sn->num_visits;
        merged_curve.num_samples = current_sn->num_visits;
        current_sn->v.curve = merged_curve;

        current_sn->v.curve *= {current_sn->common_data->gamma, current_sn->common_data->gammap};
        current_sn->v.curve += {current_sn->observed_reward, current_sn->observed_penalty};
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
    , common_data({_risk_thd, _risk_thd, _exploration_constant, _risk_exploration_ratio, _gamma, _gammap, 0, agent<S, A>::handler, {}, _use_predictor})
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
            rollout(leaf, false);
        }
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
            // spdlog::debug("number of sims: {}", i);
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
        ++common_data.num_steps;
        episode_history.push_back({root->state, a, r, common_data.sample_risk_thd});

        spdlog::debug("Play action: {}", to_string(a));
        spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

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
        common_data.num_steps = 0;
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
