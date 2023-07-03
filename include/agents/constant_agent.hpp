#pragma once

#include "agents/agent.hpp"
#include "rand.hpp"

namespace gym {

template <typename S, typename A>
class constant_agent : public agent<S, A> {
private:
    S state;
    A action;
public:
    constant_agent(A a) : agent<S, A>(), action(a) {}

    ~constant_agent() override = default;

    void play() override {
        spdlog::debug("Play action: " + std::to_string(action));
        outcome_t<S> outcome = agent<S, A>::handler.play_action(action);
        spdlog::debug("  Result: s=" + std::to_string(std::get<0>(outcome)) + ", r=" + std::to_string(std::get<1>(outcome)) + ", p=" + std::to_string(std::get<2>(outcome)));
    }

    void reset() override {
        spdlog::debug("Resetting agent");
        agent<S, A>::reset();
        state = agent<S, A>::handler.current_state();
        spdlog::debug("  Current state: " + std::to_string(state));
    }

    std::string name() const override {
        return "Constant Agent (" + std::to_string(action) + ")";
    }
};

} // namespace gym
