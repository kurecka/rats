#pragma once

#include "world.hpp"

namespace world {

class investor_env : public environment<int, size_t> {
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

    investor_env(int _initial_wealth, int _target)
    : initial_wealth(_initial_wealth)
    , target(_target)
    {}

    ~investor_env() override = default;

    std::string name() const override { return "InvestorEnv"; }

    size_t num_actions() const override { return 2; }
    std::vector<size_t> possible_actions() const override { return {RISKY, SAFE}; }
    size_t get_action(size_t i) const override { return i; }
    int current_state() const override;
    bool is_over() const override;
    outcome_t<int> play_action(size_t action) override;

    void restore_checkpoint() override;
    void make_checkpoint() override;

    void reset() override;
};

} // namespace world
