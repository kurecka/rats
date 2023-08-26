#pragma once

#include <string>
#include <vector>
#include <array>
#include <cmath>

#include "pareto_curves.hpp"
#include "utils.hpp"

namespace rats {
namespace ts {

template <typename pareto_curve>
std::tuple<float, float, float, float> mix(
    pareto_curve& c1, pareto_curve& c2,
    float uct1, float uct2,
    size_t n, float eps, float risk_thd
) {
    std::vector<std::pair<float, float>> v1(n);
    std::vector<std::pair<float, float>> v2(n);

    float lp1 = c1.min_p();
    float lp2 = c2.min_p();
    
    if (risk_thd < lp1 && risk_thd < lp2) {
        return {risk_thd, lp1 < lp2, risk_thd, -std::numeric_limits<float>::infinity()};
    }

    if (lp1 > lp2) {lp2 -= eps;}
    else {lp1 -= eps;}
    float rp1 = 1;
    float rp2 = 1 - eps;

    while (rp1 - lp1 > eps || rp2 - lp2 > eps) {
        for (size_t i = 0; i < n; ++i) {
            float p1 = lp1 + (rp1 - lp1) * i / (n - 1);
            float p2 = lp2 + (rp2 - lp2) * i / (n - 1);
            v1[i] = {c1.eval(p1, uct1), p1};
            v2[i] = {c2.eval(p2, uct2), p2};
        }

        auto [idx1, idx2] = common_tangent(v1, v2);
        if (idx1 == v1.size()) {
            bool choose1 = c1.eval(risk_thd, uct1) > c2.eval(risk_thd, uct2);
            lp1 = rp1 = risk_thd * choose1;
            lp2 = rp2 = risk_thd * !choose1;
            
        } else {
            size_t lidx1 = idx1 >= 1 ? idx1 - 1 : 0;
            size_t lidx2 = idx2 >= 1 ? idx2 - 1 : 0;
            size_t ridx1 = idx1 + 1 < n ? idx1 + 1 : n - 1;
            size_t ridx2 = idx2 + 1 < n ? idx2 + 1 : n - 1;
            lp1 = v1[lidx1].second;
            lp2 = v2[lidx2].second;
            rp1 = v1[ridx1].second;
            rp2 = v2[ridx2].second;
        }
    }

    float p1 = (lp1 + rp1) / 2;
    float p2 = (lp2 + rp2) / 2;
    float prob1;

    if (risk_thd < p1 && risk_thd < p2) {
        prob1 = p1 < p2;
    } else if (risk_thd > p1 && risk_thd > p2) {
        prob1 = p1 > p2;
    } else {
        prob1 = (risk_thd - p2) / (p1 - p2);
    }

    float v = prob1 * c1.eval(p1, uct1) + (1 - prob1) * c2.eval(p2, uct2);
    return {p1, prob1, p2, v};
}

template <typename S, typename A>
struct pareto_uct_data {
    float risk_thd;
    float sample_risk_thd;
    float exploration_constant;
    environment_handler<S, A>& handler;
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

    auto& children = node->children;

    auto [min_r, max_r] = children[0].q.curve.r_bounds();
    for (size_t i = 0; i < children.size(); ++i) {
        auto [er, ep] = children[i].q.curve.r_bounds();
        min_r = std::min(min_r, er);
        max_r = std::max(max_r, er);
    }
    if (min_r >= max_r) {
        if (min_r < 0) {
            max_r = 0.9f * min_r;
        } else {
            max_r = 1.1f * min_r;
        }
    }

    std::vector<float> uct_bonus(children.size());

    for (size_t i = 0; i < children.size(); ++i) {
        uct_bonus[i] = explore * c * static_cast<float>(
            sqrt(log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
        );
    }

    float max_v = -std::numeric_limits<float>::infinity();
    float max_thd = 0;
    size_t max_idx = 0;

    for (size_t i = 0; i < children.size(); ++i) {
        for (size_t j = i+1; j < children.size(); ++j) {
            auto [p1, prob1, p2, v] = mix(
                children[i].q.curve, children[j].q.curve,
                uct_bonus[i], uct_bonus[j],
                10, 0.01f, risk_thd
            );

            if (v > max_v) {
                max_v = v;
                if (rng::unif_float() < prob1) {
                    max_idx = i;
                    max_thd = p1;
                } else {
                    max_idx = j;
                    max_thd = p2;
                }
            }
        }
    }

    node->common_data->sample_risk_thd = max_thd;
    return node->actions[max_idx];
}

template<typename S, typename A, typename DATA, typename pareto_curve>
void descend_callback(
    state_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>*,
    A, action_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>* action,
    S, state_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>* new_state
) {
    DATA* common_data = action->common_data;
    float risk_thd = action->q.risk_thd = common_data->sample_risk_thd;
    float d = action->q.curve.derivative(risk_thd);
    float new_risk_thd = new_state->v.curve.inverse_derivative(d);
    new_state->v.risk_thd = new_risk_thd;
    common_data->sample_risk_thd = new_risk_thd;
}


template<typename S, typename A, typename DATA, typename pareto_curve>
void pareto_prop_v_value(
    state_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>* sn,
    float disc_r, float disc_p
) {
    sn->num_visits++;
    sn->v.curve.update(disc_r, disc_p);
}


template<typename S, typename A, typename DATA, typename pareto_curve>
void pareto_prop_q_value(
    action_node<S, A, DATA, pareto_value<pareto_curve>, pareto_value<pareto_curve>>* an,
    float disc_r, float disc_p
) {
    an->num_visits++;
    an->q.curve.update(disc_r, disc_p);
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
    using v_t = pareto_value<quad_pareto_curve>;
    using q_t = pareto_value<quad_pareto_curve>;
    using pareto_curve = quad_pareto_curve;
    using uct_state_t = state_node<S, A, data_t, v_t, q_t>;
    using uct_action_t = action_node<S, A, data_t, v_t, q_t>;
    
    constexpr static auto select_action_f = select_action_pareto<S, A, data_t, pareto_curve>;
    constexpr static auto descend_callback_f = descend_callback<S, A, data_t, pareto_curve>;
    constexpr static auto select_leaf_f = select_leaf<S, A, data_t, v_t, q_t, select_action_f, descend_callback_f>;
    constexpr static auto propagate_f = propagate<S, A, data_t, v_t, q_t, pareto_prop_v_value<S, A, data_t, pareto_curve>, pareto_prop_q_value<S, A, data_t, pareto_curve>>;

private:
    int max_depth;
    int num_sim;
    float risk_thd;
    float gamma;

    data_t common_data;

    std::unique_ptr<uct_state_t> root;
public:
    pareto_uct(
        environment_handler<S, A> _handler,
        int _max_depth, int _num_sim, float _risk_thd, float _gamma,
        float _exploration_constant = 5.0
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , risk_thd(_risk_thd)
    , gamma(_gamma)
    , common_data({_risk_thd, _risk_thd, _exploration_constant, agent<S, A>::handler})
    , root(std::make_unique<uct_state_t>())
    {
        reset();
    }

    void play() override {
        spdlog::debug("Play: {}", name());

        for (int i = 0; i < num_sim; i++) {
            spdlog::trace("Simulation {}", i);
            common_data.sample_risk_thd = common_data.risk_thd;
            uct_state_t* leaf = select_leaf_f(root.get(), true, max_depth);
            expand_state(leaf);
            void_rollout(leaf);
            propagate_f(leaf, gamma);
            agent<S, A>::handler.sim_reset();
        }

        common_data.sample_risk_thd = common_data.risk_thd;
        A a = select_action_pareto<S, A, data_t, pareto_curve>(root.get(), false);

        static bool logged = false;
        if (!logged) {
            spdlog::get("graphviz")->info(to_graphviz_tree(*root.get(), 9));
            logged = true;
        }

        auto [s, r, p, t] = agent<S, A>::handler.play_action(a);
        spdlog::debug("Play action: {}", a);
        spdlog::debug(" Result: s={}, r={}, p={}", s, r, p);

        uct_action_t* an = root->get_child(a);
        if (an->children.find(s) == an->children.end()) {
            an->children[s] = expand_action(an, s, r, p, t);
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
    }

    std::string name() const override {
        return "pareto_uct";
    }
};

} // namespace ts
} // namespace rats


#include "test.hpp"
