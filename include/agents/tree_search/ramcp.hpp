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


template<typename SN>
struct select_action_uct {
    size_t operator()(SN* node, bool /*explore*/) const {

        float c = node->common_data->exploration_constant;

        auto& children = node->children;
        float max_v = std::max_element(children.begin(), children.end(),
                        [](auto& l, auto& r){ return l.q.first < r.q.first; })->q.first;
        float min_v = std::min_element(children.begin(), children.end(),
                        [](auto& l, auto& r){ return l.q.first < r.q.first; })->q.first;

        size_t idxa = 0;
        float max_uct = 0, uct_value = 0;
        for (size_t i = 0; i < children.size(); ++i) {
            uct_value = ((children[i].q.first - min_v) / (max_v - min_v)) +
                c * static_cast<float>(std::sqrt(std::log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            );

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
        int _graphviz_depth = -1
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , sim_time_limit(_sim_time_limit)
    , risk_thd(_risk_thd)
    , common_data({_risk_thd, _exploration_constant, _gamma, _gammap, agent<S, A>::handler, {}})
    , graphviz_depth(_graphviz_depth)
    , root(std::make_unique<state_node_t>())
    {
        // Create the linear solvers with the GLOP backend.
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
            for (int i = 0; i < num_sim; ++i) {
                simulate(i);
            }
        }

        A a = solver.get_action(root.get(), risk_thd);

        if (graphviz_depth > 0) {
            dot_tree = to_graphviz_tree(*root.get(), graphviz_depth);
        }

        auto [s, r, p, t] = common_data.handler.play_action(a);
        spdlog::debug("Play action: {}", to_string(a));
        spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

        action_node_t* an = root->get_child(a);
        if (an->children.find(s) == an->children.end()) {
            full_expand_action(an, s, r, p, t);
        }

        risk_thd = solver.update_threshold(risk_thd, a, s);

        std::unique_ptr<state_node_t> new_root = an->get_child_unique_ptr(s);
        root = std::move(new_root);
        root->get_parent() = nullptr;
    }

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