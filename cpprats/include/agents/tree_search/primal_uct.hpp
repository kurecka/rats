#pragma once
#include "tree_search.hpp"
#include "../../string_utils.hpp"
#include <string>
#include <vector>
#include <array>

namespace rats {
namespace ts {

template <typename S, typename A>
struct primal_uct_data {
    float risk_thd;
    float sample_risk_thd;
    float exploration_constant;
    float gamma;
    float gammap;
    environment_handler<S, A>& handler;
    predictor_manager<S, A> predictor;
};


template<typename SN, bool deterministic>
struct select_action_primal {
    size_t operator()(SN* node, bool explore) const {
        float risk_thd = node->common_data->sample_risk_thd;
        float c = node->common_data->exploration_constant;

        auto& children = node->children;
        auto q = children[0].q;
        auto [min_r, min_p] = q;
        auto [max_r, max_p] = q;
        for (size_t i = 0; i < children.size(); ++i) {
            auto [er, ep] = children[i].q;
            min_r = std::min(min_r, er);
            max_r = std::max(max_r, er);
            min_p = std::min(min_p, ep);
            max_p = std::max(max_p, ep);
        }
        if (min_p >= max_p) max_p = min_p + 0.1f;

        std::vector<float> ucts(children.size());
        std::vector<float> lcts(children.size());

        for (size_t i = 0; i < children.size(); ++i) {
            ucts[i] = children[i].q.first + explore * c * (max_r - min_r) * static_cast<float>(
                sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            );
            lcts[i] = children[i].q.second - explore * c * (max_p - min_p) * static_cast<float>(
                sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            );
            if (lcts[i] < 0) lcts[i] = 0;
        }

        auto mix = greedy_mix(ucts, lcts, risk_thd);
        if (!explore) {
            std::string ucts_str = "";
            for (auto u : ucts) ucts_str += to_string(u) + ", ";
            std::string lcts_str = "";
            for (auto l : lcts) lcts_str += to_string(l) + ", ";
            spdlog::trace("ucts: {}", ucts_str);
            spdlog::trace("lcts: {}", lcts_str);
            spdlog::trace("a1: {}, p2: {}, a2: {}, thd: {}", to_string(mix.v1), mix.p2, to_string(mix.v2), risk_thd);
        }

        if constexpr (deterministic) {
            return mix.v1;
        } else {
            return mix();
        }
    }
};


/*********************************************************************
 * @brief primal uct agent
 * 
 * @tparam S State type
 * @tparam A Action type
 *********************************************************************/

template <typename S, typename A>
class primal_uct : public agent<S, A> {
    using data_t = primal_uct_data<S, A>;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;
    using action_node_t = action_node<S, A, data_t, v_t, q_t>;
    
    using select_action_t = select_action_primal<state_node_t, true>;
    using descend_callback_t = void_fn<state_node_t*, A, action_node_t*, S, state_node_t*>;
    constexpr auto static select_action_f = select_action_t();
    constexpr auto static select_leaf_f = select_leaf<state_node_t, select_action_t, descend_callback_t>;
    constexpr auto static propagate_f = propagate<state_node_t, uct_prop_v_value_prob<state_node_t>, uct_prop_q_value_prob<action_node_t>>;

private:
    int max_depth;
    int num_sim;
    int sim_time_limit;
    float risk_thd;
    float gamma;

    data_t common_data;

    std::unique_ptr<state_node_t> root;
public:
    primal_uct(
        environment_handler<S, A> _handler,
        int _max_depth, float _risk_thd, float _gamma, float _gammap = 1,
        int _num_sim = 100, int _sim_time_limit = 0,
        float _exploration_constant = 5.0
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , sim_time_limit(_sim_time_limit)
    , risk_thd(_risk_thd)
    , gamma(_gamma)
    , common_data({_risk_thd, _risk_thd, _exploration_constant, gamma, _gammap, agent<S, A>::handler, {}})
    , root(std::make_unique<state_node_t>())
    {
        reset();
    }

    void simulate(int i) {
        spdlog::trace("Simulation {}", i);
        common_data.sample_risk_thd = common_data.risk_thd;
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
            int i = 0;
            while (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count() < sim_time_limit) {
                simulate(i++);
            }
        } else {
            for (int i = 0; i < num_sim; i++) {
                simulate(i);
            }
        }

        common_data.sample_risk_thd = common_data.risk_thd;
        size_t a_idx = select_action_f(root.get(), false);
        A a = root->actions[a_idx];

        auto [s, r, p, t] = agent<S, A>::handler.play_action(a);
        spdlog::debug("Play action: {}", to_string(a));
        spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

        action_node_t* an = root->get_child(a_idx);
        if (an->children.find(s) == an->children.end()) {
            update_predictor(root.get(), a, s, r, p, t);
            full_expand_action(an);
        }

        std::unique_ptr<state_node_t> new_root = an->get_child_unique_ptr(s);
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
        return "primal_uct";
    }
};

} // namespace ts
} // namespace rats
