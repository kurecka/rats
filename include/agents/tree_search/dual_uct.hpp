#pragma once

#include "tree_search.hpp"
#include <string>
#include <vector>


namespace rats {
namespace ts {

template <typename S, typename A>
struct dual_uct_data {
    float risk_thd;
    float descent_risk_thd;
    float lambda;
    float exploration_constant;
    float gamma;
    float gammap;
    int num_steps;
    environment_handler<S, A>& handler;
    predictor_manager<S, A> predictor;
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

        for (size_t i = 0; i < children.size(); ++i) {
            uct_values[i] += explore * c * (max_v - min_v) * static_cast<float>(
                sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            );
            spdlog::debug("node: {}, i: {}", node->num_visits, children[i].num_visits);
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
        float low_p = ps[a1];
        float high_p = ps[a2];
        a1 = eps_best_actions[a1];
        a2 = eps_best_actions[a2];
        if (rng::unif_float() < p2) {
            node->common_data->descent_risk_thd = std::max(risk_thd, high_p);
            return a2;
        } else {
            node->common_data->descent_risk_thd = low_p;
            return a1;
        }
    }
};

template<typename SN>
struct select_action_dual_anal {
    size_t operator()(SN* node, bool explore) const {
        float risk_thd = node->common_data->risk_thd;
        float lambda = node->common_data->lambda;
        float c = node->common_data->exploration_constant;

        auto& children = node->children;
        
        std::vector<float> uct_values(children.size());
        for (size_t i = 0; i < children.size(); ++i) {
            uct_values[i] = children[i].q.first - lambda * children[i].q.second;
            // spdlog::debug("action: {} value: {}", i, uct_values[i]);
        }

        for (size_t i = 0; i < children.size(); ++i) {
            uct_values[i] += explore * c * static_cast<float>(
                sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            );
            // spdlog::debug("node: {}, i: {}, {}", node->num_visits, children[i].num_visits, i);
            // spdlog::debug("action: {} val with bonus: {}", i, uct_values[i]);
        }

        auto best_a = std::max_element(uct_values.begin(), uct_values.end());
        size_t best_action = std::distance(uct_values.begin(), best_a);
        float best_reward = *best_a;
        // spdlog::debug("best reward: {} best action: {}", best_reward, best_action);

        std::vector<size_t> eps_best_actions;
        int tree_depth = node->node_depth() - node->common_data->num_steps;
        const float eps_const = exp(-tree_depth) * 0.1;
        // spdlog::debug("eps_const: {}, tree_depth: {}", eps_const, tree_depth);
        for (size_t i = 0; i < children.size(); ++i) {
            // spdlog::debug("{} {}, i: {}", children[i].num_visits, children[best_action].num_visits, i);
            float eps = eps_const * (sqrt( log(children[i].num_visits + 1) / (children[i].num_visits + 1) )
                        + sqrt( log(children[best_action].num_visits + 1) / (children[best_action].num_visits + 1) ));
            // spdlog::debug("eps: {}, rew: {}", eps, uct_values[i]);
            if (fabs(uct_values[i] - best_reward) <= eps) {
                // spdlog::debug("epsaction: {}", i);
                eps_best_actions.push_back(i);
            }
        }

        size_t a1 = 0, a2 = 0;
        float low_p = 1, high_p = 0;
        for (size_t idx : eps_best_actions) {
            float p = children[idx].q.second;
            // spdlog::debug("idx: {}, p: {}", idx, p);

            if (p <= low_p) {
                low_p = p;
                a1 = idx;
            }

            if (p >= high_p) {
                high_p = p;
                a2 = idx;
            }
        }

        float p2;
        if (risk_thd <= low_p) {
            p2 = 0;
        } else if (risk_thd >= high_p) {
            p2 = 1;
        } else {
            p2 = (risk_thd - low_p) / (high_p - low_p);
        }

        // spdlog::debug("a1: {} low_p: {} a2: {} high_p: {} p2: {}", a1, low_p, a2, high_p, p2);
        if (rng::unif_float() < p2) {
            node->common_data->descent_risk_thd = std::max(risk_thd, high_p);
            return a2;
        } else {
            node->common_data->descent_risk_thd = low_p;
            return a1;
        }
    }
};

/**
 * @brief Update the nodes after a descent
 * 
 * Update risk threshold after a descent through an action and a state node.
 */
template<typename SN>
struct descend_callback {
    using state_node_t = SN;
    using S = typename SN::S;
    using action_node_t = typename SN::action_node_t;
    using A = typename SN::A;
    using DATA = typename SN::DATA;
    void operator()(state_node_t* s0, A a, action_node_t* action, S s, state_node_t* new_state) const {
            DATA* common_data = action->common_data;
            
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
    
    using select_action_t = select_action_dual_anal<state_node_t>;
    using descend_callback_t = void_fn<state_node_t*, A, action_node_t*, S, state_node_t*>;
    constexpr auto static select_action_f = select_action_t();
    constexpr auto static select_leaf_f = select_leaf<state_node_t, select_action_t, descend_callback_t>;
    constexpr auto static propagate_f = propagate<state_node_t, uct_prop_v_value<state_node_t>, uct_prop_q_value<action_node_t>>;
    // constexpr auto static propagate_f = propagate<state_node_t, uct_prop_v_value_prob<state_node_t>, uct_prop_q_value_prob<action_node_t>>;
private:
    int max_depth;
    int num_sim;
    int sim_time_limit;
    float risk_thd;
    float lr;
    float initial_lambda;
    float d_lambda = 0;
    float lambda_max = 100;

    data_t common_data;

    int graphviz_depth = -1;
    std::string dot_tree;

    std::unique_ptr<state_node_t> root;
public:
    dual_uct(
        environment_handler<S, A> _handler,
        int _max_depth, float _risk_thd, float _gamma, float _gammap = 1,
        int _num_sim = 100, int _sim_time_limit = 0,
        float _exploration_constant = 5.0, float _initial_lambda = 0, float _lr = 0.0005,
        int _graphviz_depth = 0
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , sim_time_limit(_sim_time_limit)
    , risk_thd(_risk_thd)
    , lr(_lr)
    , initial_lambda(_initial_lambda)
    , common_data({_risk_thd, _risk_thd, _initial_lambda, _exploration_constant, _gamma, _gammap, 0, agent<S, A>::handler, {}})
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
        rollout(leaf);
        propagate_f(leaf);
        agent<S, A>::handler.end_sim();

        A a = select_action_f(root.get(), false);
        action_node_t* action_node = root->get_child(a);

        // spdlog::debug("lambda: {}", common_data.lambda);
        double gradient = (action_node->q.second - risk_thd) < 0 ? -1 : 1;
        common_data.lambda += 1.0 / (i + 1.0) * gradient;
        common_data.lambda = std::max(0.0f, common_data.lambda);
        common_data.lambda = std::min(lambda_max, common_data.lambda);
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

        A a = select_action_f(root.get(), false);
        if (graphviz_depth > 0) {
            dot_tree = to_graphviz_tree(*root.get(), graphviz_depth);
        }

        auto [s, r, p, t] = agent<S, A>::handler.play_action(a);
        ++common_data.num_steps;
        spdlog::debug("Play action: {}", to_string(a));
        spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

        action_node_t* an = root->get_child(a);
        if (an->children.find(s) == an->children.end()) {
            full_expand_action(an, s, r, p, t);
        }

        common_data.risk_thd = common_data.descent_risk_thd;

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
        root->state = agent<S, A>::handler.get_current_state();
        common_data.handler.gamma = common_data.gamma;
        common_data.handler.gammap = common_data.gammap;
        common_data.num_steps = 0;
    }

    std::string name() const override {
        return "dual_uct";
    }
};

} // namespace ts
} // namespace rats
