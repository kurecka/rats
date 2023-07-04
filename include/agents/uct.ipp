namespace gym {
namespace ts {

/***************************************************
 * State node implementation
 * *************************************************/

template <typename S, typename A, typename DATA>
std::string uct_state<S, A, DATA>::to_string() const {
    std::string s = "State node: ";
    s += "R: " + std::to_string(observed_reward) + " P: " + std::to_string(observed_penalty) + "\\n";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + "\\n";
    s += "V: " + std::to_string(num_visits) + "\\n";
    s += "T: " + std::to_string(terminal);
    return s;
}

template <typename S, typename A, typename DATA>
void uct_state<S, A, DATA>::expand(DATA* _common_data) {
    common_data = _common_data;
    actions = common_data->handler.possible_actions();
    children.resize(actions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
        children[i].parent = this;
        children[i].common_data = common_data;
    }
}

template <typename S, typename A, typename DATA>
A uct_state<S, A, DATA>::select_action(float risk_thd, bool explore) {
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
            best_action = actions[i];
        }
    }

    if (explore && rng::unif_float() < 0.1f) {
        return actions[rng::unif_int(children.size())];
    } else {
        return best_action;
    }
}

template <typename S, typename A, typename DATA>
void uct_state<S, A, DATA>::propagate(uct_action<S, A, DATA>* child, float gamma) {
    if (child) {
        num_visits++;
        expected_reward += (gamma * child->expected_reward - expected_reward) / num_visits;
        expected_penalty += (gamma * child->expected_penalty - expected_penalty) / num_visits;
    }
}


/***************************************************
 * Action node implementation
 * *************************************************/

template <typename S, typename A, typename DATA>
void uct_action<S, A, DATA>::add_outcome(S s, float r, float p, bool t) {
    if (children.find(s) == children.end()) {
        children[s] = std::make_unique<uct_state<S, A, DATA>>();
    }
    children[s]->observed_reward = r;
    children[s]->observed_penalty = p;

    children[s]->parent = this;
    children[s]->terminal = t;
}

template <typename S, typename A, typename DATA>
void uct_action<S, A, DATA>::propagate(uct_state<S, A, DATA>* child, float gamma) {
    num_visits++;
    expected_reward += (gamma * (child->expected_reward + child->observed_reward) - expected_reward) / num_visits;
    expected_penalty += (gamma * (child->expected_penalty + child->observed_penalty) - expected_penalty) / num_visits;
}

template <typename S, typename A, typename DATA>
std::string uct_action<S, A, DATA>::to_string() const {
    std::string s = "Action node: ";
    s += "E[R]: " + std::to_string(expected_reward) + " E[P]: " + std::to_string(expected_penalty) + " ";
    s += "V: " + std::to_string(num_visits) + " ";
    return s;
}


/***************************************************
 * UCT implementation
 * *************************************************/

template <typename S, typename A>
void UCT<S, A>::play() {
    spdlog::debug("Running simulations");
    for (int i = 0; i < num_sim; i++) {
        spdlog::trace("Simulation " + std::to_string(i));
        uct_state<S, A, data_t>* leaf = ts.select();
        leaf->expand(&common_data);
        ts.propagate(leaf);
    }

    uct_state<S, A, data_t>* root = ts.get_root();
    A a = root->select_action(risk_thd, false);

    spdlog::trace("Play action: " + std::to_string(a));
    auto [s, r, p, e] = agent<S, A>::handler.play_action(a);
    spdlog::trace("  Result: s=" + std::to_string(s) + ", r=" + std::to_string(r) + ", p=" + std::to_string(p));
    
    root->get_child(a)->add_outcome(s, r, p, e);

    ts.descent(a, s);
}

template <typename S, typename A>
void UCT<S, A>::reset() {
    agent<S, A>::reset();
    ts.reset();
}

} // namespace ts
} // namespace gym
