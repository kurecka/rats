namespace world {
namespace ts {

/***************************************************
 * State node implementation
 * *************************************************/

template <typename S>
std::string state_node<S>::to_string() const {
    std::string s = "State node: ";
    s += "R: " + std::to_string(observed_reward) + " P: " + std::to_string(observed_penalty) + "\\n";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + "\\n";
    s += "V: " + std::to_string(num_visits) + "\\n";
    s += "T: " + std::to_string(is_terminal);
    return s;
}

template <typename S>
void state_node<S>::expand(size_t num_actions) {
    children.resize(num_actions);
    for (size_t i = 0; i < num_actions; ++i) {
        children[i].parent = this;
    }
}

template <typename S>
action_t state_node<S>::select_action(float risk_thd, bool explore) {
    // best safe action
    float best_reward = -1e9;
    action_t best_action = 0;
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
            best_action = static_cast<action_t>(i);
        }
    }

    if (explore && rng::unif_float() < 0.1f) {
        return rng::unif_int(children.size());
    } else {
        return best_action;
    }
}

template <typename S>
void state_node<S>::propagate(action_node<S>* child, float gamma) {
    if (child) {
        num_visits++;
        expected_reward += (gamma * child->expected_reward - expected_reward) / num_visits;
        expected_penalty += (gamma * child->expected_penalty - expected_penalty) / num_visits;
    }
}

template <typename S>
void state_node<S>::validate() const {

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

template <typename S>
void action_node<S>::add_outcome(S s, float r, float p, bool t) {
    children[s]->observed_reward = r;
    children[s]->observed_penalty = p;

    children[s]->parent = this;
    children[s]->is_terminal = t;
}

template <typename S>
void action_node<S>::propagate(state_node<S>* child, float gamma) {
    num_visits++;
    expected_reward += (gamma * (child->expected_reward + child->observed_reward) - expected_reward) / num_visits;
    expected_penalty += (gamma * (child->expected_penalty + child->observed_penalty) - expected_penalty) / num_visits;
}

template <typename S>
std::string action_node<S>::to_string() const {
    std::string s = "Action node: ";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + " ";
    s += "V: " + std::to_string(num_visits) + " ";
    return s;
}

template <typename S>
void action_node<S>::validate() const {
    for (const auto& [state, node] : children) {
        node->validate();
    }
}

} // namespace ts
} // namespace world
