#include "utils.hpp"

namespace gym {
namespace ts {

/***************************************************
 * State node implementation
 * *************************************************/

template <typename S, typename A, typename AN, typename DATA>
A uct_state<S, A, AN, DATA>::select_action(bool explore) {
    float risk_thd = common_data->sample_risk_thd;
    float c = common_data->exploration_constant;

    float min_r = children[0].expected_reward;
    float max_r = min_r;
    float min_p = children[0].expected_penalty;
    float max_p = min_p;
    for (size_t i = 0; i < children.size(); ++i) {
        min_r = std::min(min_r, children[i].expected_reward);
        max_r = std::max(max_r, children[i].expected_reward);
        min_p = std::min(min_p, children[i].expected_penalty);
        max_p = std::max(max_p, children[i].expected_penalty);
    }
    if (min_r >= max_r) max_r = min_r + 0.1f;
    if (min_p >= max_p) max_p = min_p + 0.1f;

    std::vector<float> ucts(children.size());
    std::vector<float> lcts(children.size());

    for (size_t i = 0; i < children.size(); ++i) {
        ucts[i] = children[i].expected_reward + explore * c * (max_r - min_r) * static_cast<float>(
            sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001))
        );
        lcts[i] = children[i].expected_penalty - explore * c * (max_p - min_p) * static_cast<float>(
            sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001))
        );
        if (lcts[i] < 0) lcts[i] = 0;
    }

    auto [a1, p2, a2] = greedy_mix(ucts, lcts, risk_thd);

    return actions[a1];
    // } else {
    //     if (rng::unif_float() < p2) {
    //         return actions[a2];
    //     } else {
    //         return actions[a1];
    //     }
    // }
}

} // namespace ts
} // namespace gym
