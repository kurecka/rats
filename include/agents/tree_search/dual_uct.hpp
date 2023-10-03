#pragma once

#include "tree_search.hpp"
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
    environment_handler<S, A>& handler;
};


template<typename SN>
struct select_action_dual {
    size_t operator()(SN* node, bool explore) const {
        float risk_thd = node->common_data->risk_thd;
        float lambda = node->common_data->lambda;
        float c = node->common_data->exploration_constant;

        auto& children = node->children;
        
        float min_v = children[0].q.first - lambda * children[0].q.second;
        float max_v = min_v;
        std::vector<float> uct_values(children.size());
        for (size_t i = 0; i < children.size(); ++i) {
            float val = children[i].q.first - lambda * children[i].q.second;
            min_v = std::min(min_v, val);
            max_v = std::max(max_v, val);
            uct_values[i] = val;
        }
        if (min_v < 0) {
            max_v = 0.9f * min_v;
        } else {
            max_v = 1.1f * min_v;
        }

        for (size_t i = 0; i < children.size(); ++i) {
            uct_values[i] += explore * c * (max_v - min_v) * static_cast<float>(
                sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            );
        }

        float best_reward = *std::max_element(uct_values.begin(), uct_values.end());
        float eps = (max_v - min_v) * 0.1f;
        std::vector<size_t> eps_best_actions;
        for (size_t i = 0; i < children.size(); ++i) {
            if (uct_values[i] >= best_reward - eps) {
                eps_best_actions.push_back(i);
            }
        }
        
        std::vector<float> rs(eps_best_actions.size());
        for (size_t idx : eps_best_actions) {
            rs[idx] = children[idx].q.first;
        }

        std::vector<float> ps(eps_best_actions.size());
        for (size_t idx : eps_best_actions) {
            ps[idx] = children[idx].q.second;
        }

        auto [a1, p2, a2] = greedy_mix(rs, ps, risk_thd);
        a1 = eps_best_actions[a1];
        a2 = eps_best_actions[a2];
        if (rng::unif_float() < p2) {
            return a2;
        } else {
            return a1;
        }
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

template <typename S, typename A>
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
    float risk_thd;
    float lr;
    float initial_lambda;
    float d_lambda = 0;

    data_t common_data;

    std::unique_ptr<state_node_t> root;
public:
    dual_uct(
        environment_handler<S, A> _handler,
        int _max_depth, int _num_sim, float _risk_thd, float _gamma,
        float _exploration_constant = 5.0, float _initial_lambda = 0, float _lr = 0.0005
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , risk_thd(_risk_thd)
    , lr(_lr)
    , initial_lambda(_initial_lambda)
    , common_data({_risk_thd, _initial_lambda, _exploration_constant, _gamma, agent<S, A>::handler})
    , root(std::make_unique<state_node_t>())
    {
        reset();
    }

    void play() override {
        spdlog::debug("Play: {}", name());

        for (int i = 0; i < num_sim; i++) {
            spdlog::trace("Simulation {}", i);
            spdlog::trace("Select leaf");
            state_node_t* leaf = select_leaf_f(root.get(), true, max_depth);
            spdlog::trace("Expand leaf");
            expand_state(leaf);
            spdlog::trace("Rollout");
            rollout(leaf);
            spdlog::trace("Propagate");
            propagate_f(leaf, common_data.gamma);
            agent<S, A>::handler.end_sim();

            spdlog::trace("Select action");
            A a = select_action_f(root.get(), false);
            action_node_t* action_node = root->get_child(a);

            spdlog::trace("Update lambda");
            float grad = action_node->q.second - common_data.risk_thd;
            // d_lambda += (grad - d_lambda) * 0.2f;
            d_lambda = grad;
            common_data.lambda += lr * d_lambda;
            common_data.lambda = std::max(0.0f, common_data.lambda);
            // spdlog::info("Lambda: {}", common_data.lambda);
        }

        A a = select_action_f(root.get(), false);

        // static bool logged = false;
        // if (!logged) {
        //     spdlog::get("graphviz")->info(to_graphviz_tree(*root.get(), 9));
        //     logged = true;
        // }

        auto [s, r, p, t] = agent<S, A>::handler.play_action(a);
        spdlog::debug("Play action: {}", a);
        spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

        action_node_t* an = root->get_child(a);
        if (an->children.find(s) == an->children.end()) {
            expand_action(an, s, r, p, t);
        }

        common_data.risk_thd = an->children[s]->v.second;

        std::unique_ptr<state_node_t> new_root = an->get_child_unique_ptr(s);
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
    }

    std::string name() const override {
        return "dual_uct";
    }
};

} // namespace ts
} // namespace rats
