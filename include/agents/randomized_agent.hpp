#pragma once

#include "agents/agent.hpp"
#include "string_utils.hpp"
#include "rand.hpp"


namespace rats {

template <typename S, typename A>
class randomized_agent : public agent<S, A> {
private:
    S state;
public:
    randomized_agent(environment_handler<S, A> _handler) : agent<S, A>(_handler) {}

    ~randomized_agent() override = default;

    void play() override {
        A action = agent<S, A>::handler.get_action(
            rng::unif_int(agent<S, A>::handler.num_actions())
        );
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
        return "Randomized Agent";
    }
};

} // namespace rats
