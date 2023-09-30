#pragma once

#include "agents/agent.hpp"
#include "../string_utils.hpp"
#include "rand.hpp"


namespace rats {

template <typename S, typename A>
class constant_agent : public agent<S, A> {
private:
    S state;
    A action;
public:
    constant_agent(environment_handler<S, A> _handler, A a)
    : agent<S, A>(_handler), action(a) {} 

    ~constant_agent() override = default;

    void play() override {
        spdlog::debug("Play action: " + to_string(action));
        outcome_t<S> outcome = agent<S, A>::handler.play_action(action);
        spdlog::debug("  Result: s=" + to_string(std::get<0>(outcome)) + ", r=" + to_string(std::get<1>(outcome)) + ", p=" + to_string(std::get<2>(outcome)));
    }

    void reset() override {
        spdlog::debug("Resetting agent");
        agent<S, A>::reset();
        state = agent<S, A>::handler.get_current_state();
        spdlog::debug("  Current state: " + to_string(state));
    }

    std::string name() const override {
        return "Constant Agent (" + to_string(action) + ")";
    }
};

} // namespace rats
