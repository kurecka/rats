#pragma once

#include "world.hpp"
#include "rand.hpp"

namespace world {

template <typename S>
class randomized_agent : public agent<S> {
private:
    S state;
public:
    // randomized_agent(environment_handler<S> handler) : agent<S>(handler) {}
    randomized_agent() : agent<S>() {}

    ~randomized_agent() override = default;

    void play() override {
        action_t action = static_cast<action_t>(rng::unif_int(static_cast<int>(agent<S>::handler.num_actions())));
        spdlog::debug("Play action: " + std::to_string(action));
        outcome_t<S> outcome = agent<S>::handler.play_action(action);
        spdlog::debug("  Result: s=" + std::to_string(std::get<0>(outcome)) + ", r=" + std::to_string(std::get<1>(outcome)) + ", p=" + std::to_string(std::get<2>(outcome)));
    }

    void reset() override {
        spdlog::debug("Resetting agent");
        agent<S>::reset();
        state = agent<S>::handler.current_state();
        spdlog::debug("  Current state: " + std::to_string(state));
    }

    std::string name() const override {
        return "Randomized Agent";
    }
};


template <typename S>
class constant_agent : public agent<S> {
private:
    S state;
    action_t action;
public:
    constant_agent(action_t a) : agent<S>(), action(a) {}

    ~constant_agent() override = default;

    void play() override {
        spdlog::debug("Play action: " + std::to_string(action));
        outcome_t<S> outcome = agent<S>::handler.play_action(action);
        spdlog::debug("  Result: s=" + std::to_string(std::get<0>(outcome)) + ", r=" + std::to_string(std::get<1>(outcome)) + ", p=" + std::to_string(std::get<2>(outcome)));
    }

    void reset() override {
        spdlog::debug("Resetting agent");
        agent<S>::reset();
        state = agent<S>::handler.current_state();
        spdlog::debug("  Current state: " + std::to_string(state));
    }

    std::string name() const override {
        return "Constant Agent (" + std::to_string(action) + ")";
    }
};

} // namespace world
