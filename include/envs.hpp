#pragma once

#include "world.hpp"

namespace world {

class investor_env : public environment<int> {
private:
    int initial_wealth;
    int wealth;
    int target;
    int checkpoint;
public:

    enum investor_action {
        RISKY = 0,
        SAFE = 1
    };

    investor_env(int initial_wealth, int target)
    : initial_wealth(initial_wealth)
    , target(target)
    {}

    ~investor_env() = default;

    std::string name() const override { return "InvestorEnv"; };

    int num_actions() const override { return 2; };
    int current_state() const override;
    bool is_over() const override;
    outcome_t<int> play_action(action_t action) override;

    void restore_checkpoint() override;
    void make_checkpoint() override;

    void reset() override;
};

} // namespace world