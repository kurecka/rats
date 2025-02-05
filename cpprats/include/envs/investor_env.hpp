#pragma once

#include <map>
#include "envs/env.hpp"

namespace rats { 

class investor_env : public environment<int, size_t> {
private:
    int initial_wealth;
    int wealth;
    int target;
    int checkpoint;

    std::map<size_t, int> checkpoints;
public:

    enum investor_action {
        RISKY = 0,
        SAFE = 1
    };

    investor_env(int _initial_wealth, int _target)
    : initial_wealth(_initial_wealth)
    , wealth(_initial_wealth)
    , target(_target)
    , checkpoint(_initial_wealth)
    , checkpoints(std::map<size_t, int>())
    {}

    ~investor_env() override = default;

    std::string name() const override { return "InvestorEnv"; }
    ConstraintType get_constraint_type() const override { return ConstraintType::RISK; }

    // TODO: not supported for now
    std::pair<float, float> get_expected_reward( int, size_t, int ) const override { return {0, 0}; }
    bool is_terminal( int ) const override {
        return (wealth == target) || (wealth <= 0);
    }

    std::pair<float, float> reward_range() const override { return {-2, 12}; }
    std::pair<float, float> penalty_range() const override { return {0, 1}; }
    size_t num_actions() const override { return 2; }
    std::vector<size_t> possible_actions(int = {}) const override { return {RISKY, SAFE}; }
    size_t get_action(size_t i) const override { return i; }
    int current_state() const override;
    bool is_over() const override;
    outcome_t<int> play_action(size_t action) override;

    void restore_checkpoint(size_t id) override;
    void make_checkpoint(size_t id) override;
    std::map<int, float> outcome_probabilities(int state, size_t action) const override;

    void reset() override;
};

} // namespace rats
