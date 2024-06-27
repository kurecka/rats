#include "envs/investor_env.hpp"

namespace rats {

int investor_env::current_state() const {
    return wealth;
}

bool investor_env::is_over() const {
    return wealth >= target || wealth <= 0;
}

outcome_t<int> investor_env::play_action(size_t action) {
    if (action == RISKY) {
        if (rng::bernoulli(0.2)) {
            wealth += 12;
        } else {
            wealth -= 2;
        }
    } else {
        if (rng::bernoulli(0.7)) {
            wealth += 1;
        } else {
            wealth -= 1;
        }
    }
    
    float reward = -1;

    float punishment = 0;
    if (wealth <= 0) {
        punishment = 1;
    }

    bool over = wealth <= 0 || wealth >= target;

    return {wealth, reward, punishment, over};
}

void investor_env::make_checkpoint(size_t id) {
    if (id == 0) {
        checkpoint = wealth;
    } else {
        checkpoints[id] = wealth;
    }
}

void investor_env::restore_checkpoint(size_t id) {
    if (id == 0) {
        wealth = checkpoint;
    } else {
        wealth = checkpoints[id];
    }
}

std::map<int, float> investor_env::outcome_probabilities(int state, size_t action) const {
    std::map<int, float> probs;
    if (action == RISKY) {
        probs[state + 12] = 0.2f;
        probs[state - 2] = 0.8f;
    } else {
        probs[state + 1] = 0.7f;
        probs[state - 1] = 0.3f;
    }
    return probs;
}

void investor_env::reset() {
    spdlog::debug("Reset environment");
    wealth = initial_wealth;
}
} // namespace rats
