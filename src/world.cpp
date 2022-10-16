#include "world.hpp"

namespace world {

std::vector<int> investor_env::get_action_space() {
    return {
        investor_action::RISKY,
        investor_action::SAFE
        };
}

int investor_env::get_state() {
    return wealth;
}

bool investor_env::is_over() {
    return wealth >= target || wealth <= 0;
}

std::tuple<int, float, float, bool> investor_env::play_action(int action, int) {
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
    wealth = initial_wealth;
}

} // namespace world