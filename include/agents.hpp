#pragma once

#include "world.hpp"

namespace world {

template <typename S>
class randomized_agent : public agent<S> {
private:
    S state;
public:
    // randomized_agent(environment_handler<S> handler) : agent<S>(handler) {}
    randomized_agent() : agent<S>() {}

    ~randomized_agent() = default;

    void play() override {
        int action = unif_int(agent<S>::handler.num_actions());
        logger.debug("Play action: " + std::to_string(action));
        outcome_t<S> outcome = agent<S>::handler.play_action(action);
        logger.debug("  Result: s=" + std::to_string(std::get<0>(outcome)) + ", r=" + std::to_string(std::get<1>(outcome)) + ", p=" + std::to_string(std::get<2>(outcome)));
    }

    void reset() override {
        logger.debug("Resetting agent");
        agent<S>::reset();
        state = agent<S>::handler.current_state();
        logger.debug("  Current state: " + std::to_string(state));
    }

    std::string name() const override {
        return "Randomized Agent";
    }
};

} // namespace world