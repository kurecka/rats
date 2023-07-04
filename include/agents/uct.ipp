namespace gym {
namespace ts {

/***************************************************
 * State node implementation
 * *************************************************/

template <typename S, typename A>
std::string state_node<S, A>::to_string() const {
    std::string s = "State node: ";
    s += "R: " + std::to_string(observed_reward) + " P: " + std::to_string(observed_penalty) + "\\n";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + "\\n";
    s += "V: " + std::to_string(num_visits) + "\\n";
    s += "T: " + std::to_string(is_terminal);
    return s;
}

template <typename S, typename A>
void state_node<S, A>::expand(size_t num_actions) {
    children.resize(num_actions);
    for (size_t i = 0; i < num_actions; ++i) {
        children[i].parent = this;
    }
}

template <typename S, typename A>
A state_node<S, A>::select_action(float risk_thd, bool explore) {
    // best safe action
    float best_reward = -1e9;
    A best_action = 0;
    float best_penalty = 1;
    float min_r = children[0].expected_reward;
    float max_r = min_r;
    for (size_t i = 0; i < children.size(); ++i) {
        min_r = std::min(min_r, children[i].expected_reward);
        max_r = std::max(max_r, children[i].expected_reward);
    }
    if (min_r >= max_r) max_r += 1;

    for (size_t i = 0; i < children.size(); ++i) {
        float ucb = children[i].expected_reward + explore * 5 * (max_r - min_r) * static_cast<float>(
            sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001))
        );
        float lcb = children[i].expected_penalty - explore * 5 * static_cast<float>(
            sqrt(log(num_visits + 1) / (children[i].num_visits + 0.0001))
        );

        if ((ucb > best_reward && lcb < best_penalty) || (best_penalty > risk_thd && lcb < best_penalty)) {
            best_reward = ucb;
            best_penalty = lcb;
            best_action = static_cast<A>(i);
        }
    }

    if (explore && rng::unif_float() < 0.1f) {
        return rng::unif_int(children.size());
    } else {
        return best_action;
    }
}

template <typename S, typename A>
void state_node<S, A>::propagate(action_node<S, A>* child, float gamma) {
    if (child) {
        num_visits++;
        expected_reward += (gamma * child->expected_reward - expected_reward) / num_visits;
        expected_penalty += (gamma * child->expected_penalty - expected_penalty) / num_visits;
    }
}

template <typename S, typename A>
void state_node<S, A>::validate() const {

    int s = 0;
    for (const auto& c : children) {
        s += c.num_visits;
    }
    assert(s == num_visits);

    for (const auto& c : children) {
        c.validate();
    }
}

/***************************************************
 * Action node implementation
 * *************************************************/

template <typename S, typename A>
void action_node<S, A>::add_outcome(S s, float r, float p, bool t) {
    if (children.find(s) == children.end()) {
        children[s] = std::make_unique<state_node<S, A>>();
    }
    children[s]->observed_reward = r;
    children[s]->observed_penalty = p;

    children[s]->parent = this;
    children[s]->is_terminal = t;
}

template <typename S, typename A>
void action_node<S, A>::propagate(state_node<S, A>* child, float gamma) {
    num_visits++;
    expected_reward += (gamma * (child->expected_reward + child->observed_reward) - expected_reward) / num_visits;
    expected_penalty += (gamma * (child->expected_penalty + child->observed_penalty) - expected_penalty) / num_visits;
}

template <typename S, typename A>
std::string action_node<S, A>::to_string() const {
    std::string s = "Action node: ";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + " ";
    s += "V: " + std::to_string(num_visits) + " ";
    return s;
}

template <typename S, typename A>
void action_node<S, A>::validate() const {
    for (const auto& [state, node] : children) {
        node->validate();
    }
}

} // namespace ts
} // namespace gym
