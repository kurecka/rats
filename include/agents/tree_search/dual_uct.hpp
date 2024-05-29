#pragma once

#include "tree_search.hpp"
#include "string_utils.hpp"
#include "lp.hpp"
#include <string>
#include <vector>


namespace rats {
namespace ts {

template <typename S, typename A>
struct dual_uct_data {
    float risk_thd;
    float lambda;
    float exploration_constant;
    float gamma;
    float gammap;
    int num_steps;
    environment_handler<S, A>& handler;
    predictor_manager<S, A> predictor;
    mixture mix;
};


template<typename SN>
struct select_action_dual {
    size_t operator()(SN* node, bool explore) const {
        float risk_thd = node->common_data->risk_thd;
        float lambda = node->common_data->lambda;
        float c = node->common_data->exploration_constant;

        auto& children = node->children;
        
        std::vector<float> uct_values(children.size());
        for (size_t i = 0; i < children.size(); ++i) {
            uct_values[i] = children[i].q.first - lambda * children[i].q.second;
        }

        // get max of uct_values
        float max_v = *std::max_element(uct_values.begin(), uct_values.end());
        float min_v = *std::min_element(uct_values.begin(), uct_values.end());
        
        if (max_v <= min_v) {
            max_v = min_v + 1;
        }

        for (size_t i = 0; i < children.size(); ++i) {
            uct_values[i] += explore * (max_v - min_v) * c * static_cast<float>(
                sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            );
        }

        auto best_a = std::max_element(uct_values.begin(), uct_values.end());
        size_t best_action = std::distance(uct_values.begin(), best_a);
        float best_reward = *best_a;

        std::vector<size_t> eps_best_actions;
        int tree_depth = node->node_depth() - node->common_data->num_steps;
        const float eps_const = exp(-tree_depth) * 0.1;
        for (size_t i = 0; i < children.size(); ++i) {
            float eps = eps_const * (sqrt( log(children[i].num_visits + 1) / (children[i].num_visits + 1) )
                        + sqrt( log(children[best_action].num_visits + 1) / (children[best_action].num_visits + 1) ));
            if (fabs(uct_values[i] - best_reward) <= eps) {
                eps_best_actions.push_back(i);
            }
        }


        size_t low_action = eps_best_actions.front(), high_action = eps_best_actions.front();
        float low_penalty = 1, high_penalty = 0;
        for (size_t idx : eps_best_actions) {
            auto& [reward, penalty] = children[idx].q;
            if (penalty <= low_penalty) { low_penalty = penalty; low_action = idx; }
            if (penalty >= high_penalty) { high_penalty = penalty; high_action = idx; }
        }

        node->common_data->mix = mixture(low_action, high_action, low_penalty, high_penalty, risk_thd);
        return node->common_data->mix.sample();
    }
};

/*********************************************************************
 * @brief dual uct agent
 * 
 * @tparam S State type
 * @tparam A Action type
 * 
 * @note Lambda is preserved between epochs
 * @note Lambda is updated by the following rule:
 * d_lambda = (1-alpha) * d_lambda) + alpha * (sample_risk - risk_thd)
 * lambda = lambda + lr * d_lambda
 *********************************************************************/

template <typename S, typename A, bool use_lp = false>
class dual_uct : public agent<S, A> {
    using data_t = dual_uct_data<S, A>;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;
    using action_node_t = action_node<S, A, data_t, v_t, q_t>;
    
    using select_action_t = select_action_dual<state_node_t>;
    using descend_callback_t = void_fn<state_node_t*, A, action_node_t*, S, state_node_t*>;
    constexpr auto static select_action_f = select_action_t();
    constexpr auto static select_leaf_f = select_leaf<state_node_t, select_action_t, descend_callback_t>;
    constexpr auto static propagate_f = propagate<state_node_t, uct_prop_v_value<state_node_t>, uct_prop_q_value<action_node_t>>;
private:
    int max_depth;
    int num_sim;
    int sim_time_limit;
    float risk_thd;
    float lr;
    bool use_rollout;
    float initial_lambda;
    float d_lambda = 0;
    float lambda_max = 100;

    data_t common_data;

    int graphviz_depth = -1;
    std::string dot_tree;

    std::unique_ptr<state_node_t> root;
    lp::tree_lp_solver<state_node_t> solver;
    
public:
    dual_uct(
        environment_handler<S, A> _handler,
        int _max_depth, float _risk_thd, float _gamma, float _gammap = 1,
        int _num_sim = 100, int _sim_time_limit = 0,
        float _exploration_constant = 5.0, float _initial_lambda = 2, float _lr = -1,
        bool _rollout = true,
        int _graphviz_depth = 0
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , sim_time_limit(_sim_time_limit)
    , risk_thd(_risk_thd)
    , lr(_lr)
    , use_rollout(_rollout)
    , initial_lambda(_initial_lambda)
    , common_data({_risk_thd, _initial_lambda, _exploration_constant, _gamma, _gammap, 0, agent<S, A>::handler})
    , graphviz_depth(_graphviz_depth)
    , root(std::make_unique<state_node_t>())
    {
        reset();
    }

    std::string get_graphviz() const {
        return dot_tree;
    }

    void simulate(int i) {
        state_node_t* leaf = select_leaf_f(root.get(), true, max_depth);
        expand_state(leaf);
        if (use_rollout) {
            rollout(leaf, true);
        }
        propagate_f(leaf);
        agent<S, A>::handler.end_sim();
        size_t a_idx = select_action_f(root.get(), false);
        action_node_t* action_node = root->get_child(a_idx);

        double gradient = (action_node->q.second - common_data.risk_thd) < 0 ? -1 : 1;
        float alpha = 1.0 / (i + 1.0);
        if (lr > 0) {
            alpha = std::min(alpha, lr);
        }
        common_data.lambda += alpha * gradient;
        common_data.lambda = std::max(0.0f, common_data.lambda);
        common_data.lambda = std::min(lambda_max, common_data.lambda);
    }

    void play() override {
        spdlog::debug("Play: {}", name());
        spdlog::debug("thd {}", common_data.risk_thd);
        spdlog::debug("lambda {}", common_data.lambda);

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

        if (graphviz_depth > 0) {
            dot_tree = to_graphviz_tree(*root.get(), graphviz_depth);
        }

        std::unique_ptr<state_node_t> new_root;
        if constexpr (use_lp) {
            A a = solver.get_action(root.get(), common_data.risk_thd);
            auto [s, r, p, t] = common_data.handler.play_action(a);
            spdlog::debug("Play action: {}", to_string(a));
            spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

            // spdlog::info("Steps: {}, Action: {}, State: {}, Reward: {}", common_data.num_steps, to_string(a), to_string(s), r);

            action_node_t* an = root->get_child(a);
            if (an->children.find(s) == an->children.end()) {
                update_predictor(root.get(), a, s, r, p, t);
                full_expand_action(an);
            }

            new_root = an->get_child_unique_ptr(s);
            common_data.risk_thd = solver.update_threshold(common_data.risk_thd, a, s, p);
        } else {
            size_t a_idx = select_action_f(root.get(), false);
            A a = root->actions[a_idx];
            auto [s, r, p, t] = agent<S, A>::handler.play_action(a);
            spdlog::debug("Play action: {}", to_string(a));
            spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

            // spdlog::info("Steps: {}, Action: {}, State: {}, Reward: {}", common_data.num_steps, to_string(a), to_string(s), r);

            action_node_t* an = root->get_child(a_idx);
            if (an->children.find(s) == an->children.end()) {
                update_predictor(root.get(), a, s, r, p, t);
                full_expand_action(an);
            }

            // admissable cost
            float immediate_penalty = 0;
            auto outcomes = common_data.predictor.predict_probs(root->state, a);
            auto signals = common_data.predictor.predict_signals(root->state, a);
            for (auto& [outcome_state, outcome_prob] : outcomes) {
                float observed_penalty = std::get<1>(signals[outcome_state]);
                immediate_penalty += outcome_prob * observed_penalty;
            }

            new_root = an->get_child_unique_ptr(s);
            common_data.risk_thd = std::max(common_data.mix.update_thd(common_data.risk_thd, immediate_penalty), 0.0f);
        }
        ++common_data.num_steps;
        root = std::move(new_root);
        root->get_parent() = nullptr;
    }


    void reset() override {
        spdlog::debug("Reset: {}", name());
        agent<S, A>::reset();
        common_data.risk_thd = risk_thd;
        common_data.lambda = initial_lambda;
        d_lambda = 0;
        root = std::make_unique<state_node_t>();
        root->common_data = &common_data;
        root->state = agent<S, A>::handler.get_current_state();
        common_data.handler.gamma = common_data.gamma;
        common_data.handler.gammap = common_data.gammap;
        common_data.num_steps = 0;
    }

    std::string name() const override {
        if constexpr (use_lp) {
            return "dual_ramcp";
        } else {
            return "dual_uct";
        }
    }
};

} // namespace ts
} // namespace rats
