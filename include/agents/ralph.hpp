#pragma once

#include "gym.hpp"
#include "tree_search.hpp"
#include "rand.hpp"
#include <cmath>

namespace gym {
namespace st {

struct A_DATA {
    int num_visits;
    float average_payoff;
    float average_penalty;
};

template <typename A>
struct node_data {
    std::map<A, A_DATA> action_data;
};

template <typename S, typename A>
class ralph : public search_tree<S, A, node_data<A>> {
    float exploration_constant;
public:
    using node_data_t = node_data<A>;

    ralph(environment<S, A>* env, float exploration_constant = 0.5) : search_tree<S, A, node_data_t>(env), exploration_constant(exploration_constant) {};
    const A& UCB_action(node<S, A, node_data_t> *nod, bool greedy = false);

    const A& select_action(node<S, A, node_data_t> *nod) override;
    const A& explore_action(node<S, A, node_data_t> *nod) override;
    std::tuple<float, float, std::vector<float>> predict(node<S, A, node_data_t>* nod, const A& action, const S& state) override;
};

template <typename S, typename A>
const A& ralph<S, A>::UCB_action(node<S, A, node_data_t> *nod, bool greedy) {
    float min_v, max_v;
    min_v = max_v = nod->a_data.begin()->second.average_payoff;
    for (auto& [a, data] : nod->a_data) {
        min_v = std::min(min_v, data.average_payoff);
        max_v = std::min(max_v, data.average_payoff);
    }

    // Compute UCB values
    std::vector<float> ucb_values;
    float ucb_sum = 0;
    for (auto& [action, data] : nod->a_data) {
        float base = 0;
        if (min_v < max_v) {
            base = (data.average_payoff - min_v) / (max_v - min_v);
        }
        float ucb = base + exploration_constant * data.prob_prediction * sqrt(log(nod->num_visits) / (1 + data.num_visits));
        ucb_values.push_back(ucb);
        ucb_sum += ucb;
    }

    // Normalize UCB values and compute argmax
    float max_ucb = -1;
    float max_index = 0;
    for (int i = 0; i < ucb_values.size(); i++) {
        ucb_values[i] /= ucb_sum;
        if (ucb_values[i] > max_ucb) {
            max_ucb = ucb_values[i];
            max_index = i;
        }
    }

    int action_index;
    if (greedy) {
        action_index = max_index;
    } else {
        float rand_val = unif_float();
        float sum = 0;
        for (int i = 0; i < ucb_values.size(); i++) {
            sum += ucb_values[i];
            if (sum > rand_val) {
                action_index = i;
                break;
            }
        }
    }

    // Select action
    
    return (nod->a_data.begin() + action_index)->first;
}

template <typename S, typename A>
const A& ralph<S, A>::select_action(node<S, A, node_data_t> *nod) {
    return UCB_action(nod, true);
}

template <typename S, typename A>
const A& ralph<S, A>::explore_action(node<S, A, node_data_t> *nod) {
    return UCB_action(nod, false);
}

template <typename S, typename A>
std::tuple<float, float, std::vector<float>> ralph<S, A>::predict(node<S, A, node_data_t>*, const A&, const S&) {
    float payoff_prediction = 0;
    float penalty_prediction = 0;
    std::vector<float> probs(this->env->get_action_space().size());

    return {payoff_prediction, penalty_prediction, probs};
}

}   // namespace st
}   // namespace gym
