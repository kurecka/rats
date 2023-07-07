namespace gym {
namespace ts {

/***************************************************
 * State node implementation
 * *************************************************/

template <typename S, typename A, typename AN, typename DATA>
A uct_state<S, A, AN, DATA>::select_action(bool explore) {
    float lambda = common_data->lambda;
    float c = common_data->exploration_constant;
    
    float min_v = children[0].expected_reward - lambda * children[0].expected_penalty;
    float max_v = min_v;
    std::vector<float> uct_values(children.size());
    for (size_t i = 0; i < children.size(); ++i) {
        float val = children[i].expected_reward - lambda * children[i].expected_penalty;
        min_v = std::min(min_v, val);
        max_v = std::max(max_v, val);
        uct_values[i] = val;
    }
    if (min_v >= max_v) max_v = min_v + 1;

    for (size_t i = 0; i < children.size(); ++i) {
        uct_values[i] += explore * c * (max_v - min_v) * static_cast<float>(
            sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001))
        );
    }

    size_t best_action = 0;
    float best_reward = uct_values[0];
    for (size_t i = 1; i < children.size(); ++i) {
        if (uct_values[i] > best_reward) {
            best_reward = uct_values[i];
            best_action = i;
        }
    }

    return actions[best_action];
}

} // namespace ts
} // namespace gym
