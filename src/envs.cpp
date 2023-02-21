#include "envs.hpp"

namespace world {

int investor_env::current_state() const {
    return wealth;
}

bool investor_env::is_over() const {
    return wealth >= target || wealth <= 0;
}

outcome_t<int> investor_env::play_action(action_t action) {
    if (action == RISKY) {
        if (bernoulli(0.2)) {
            wealth += 12;
        } else {
            wealth -= 2;
        }
    } else {
        if (bernoulli(0.7)) {
            wealth += 1;
        } else {
            wealth -= 1;
        }
    }
    
    float reward = -0.1;
    if (wealth >= target) {
        reward = 1;
    }

    float punishment = 0;
    if (wealth <= 0) {
        punishment = 1;
    }

    bool over = wealth <= 0 || wealth >= target;

    return {wealth, reward, punishment, over};
}

void investor_env::make_checkpoint() {
    checkpoint = wealth;
}

void investor_env::restore_checkpoint() {
    wealth = checkpoint;
}

void investor_env::reset() {
    logger.debug("Reset environment");
    wealth = initial_wealth;
}

} // namespace world