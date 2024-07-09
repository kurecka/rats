#pragma once

#include <map>
#include "envs/env.hpp"
#include "rand.hpp"

namespace rats { 

class ccpomcp_ex2 : public environment<int, size_t> {
private:
    int state;
    int length;
    float small_reward;
    float small_risk;
    float large_reward;
    float large_risk;

    std::map<size_t, int> checkpoints;
public:

    enum actions {
        STAY=0, GO=1
    };

    ccpomcp_ex2(int length, float small_reward, float small_risk, float large_reward, float large_risk)
    : length(length)
    , small_reward(small_reward)
    , small_risk(small_risk)
    , large_reward(large_reward)
    , large_risk(large_risk)
    {
        reset();
    }

    ~ccpomcp_ex2() override = default;

    std::string name() const override { return "CC-POMCP Ex2"; }
    ConstraintType get_constraint_type() const override { return ConstraintType::RISK; }

    std::pair<float, float> reward_range() const override { return {0, 50}; }
    std::pair<float, float> penalty_range() const override { return {0, 1}; }
    size_t num_actions() const override { return 2; }
    std::vector<size_t> possible_actions(int = {}) const override { return { STAY, GO }; }
    size_t get_action(size_t i) const override { return i; }
    int current_state() const override { return state; }
    bool is_over() const override { return state < -2; }

    // TODO : not supported for now
    bool is_terminal( int state ) const override { return state == -4; }
    std::pair< float, float > get_expected_reward( int state , size_t action, int succ ) const override {
        return {0, 0};
    }

    outcome_t<int> play_action(size_t action) override {
        if (state == 0) {
            if (action == STAY) {
                state = -1;
                return {-1, 0, 0, false};
            } else {
                state++;
                return {state, 0, 0, false};
            }
        } else if (state > 0 && state < length) {
            if (action == STAY) {
                state = -3;
                return {-3, 0, 0, true};
            } else {
                state++;
                return {state, 0, 0, false};
            }
        } else if (state == length) {
            if (action == STAY) {
                state = -2;
                return {-2, 0, 0, false};
            } else {
                state = -3;
                return {-3, 0, 0, true};
            }
        } else if (state == -1) {
            if (action == STAY) {
                if (rng::unif_float() < 0.5) {
                    state = -3;
                    return {-3, 0.1, 0, true};
                } else {
                    state = -4;
                    return {-4, 0.1, 1, true};
                }
            } else {
                if (rng::unif_float() > small_risk) {
                    state = -3;
                    return {-3, small_reward, 0, true};
                } else {
                    state = -4;
                    return {-4, small_reward, 1, true};
                }
            }
        } else if (state == -2) {
            if (action == STAY) {
                state = -3;
                return {-3, 0, 0, true};
            } else {
                if (rng::unif_float() > large_risk) {
                    state = -3;
                    return {-3, large_reward, 0, true};
                } else {
                    state = -4;
                    return {-4, large_reward, 1, true};
                }
            }
        } else {
            return {-4, 0, 1, true};
        }
    }

    void restore_checkpoint(size_t id) override {
        state = checkpoints[id];
    }
    void make_checkpoint(size_t id) override {
        checkpoints[id] = state;
    }
    std::map<int, float> outcome_probabilities(int state, size_t action) const override {
        if (state >= 0 && state < length) {
            if (action == STAY) {
                return {{-1, 1}};
            } else {
                return {{state+1, 1}};
            }
        } else if (state == length) {
            if (action == STAY) {
                return {{-1, 1}};
            } else {
                return {{-2, 1}};
            }
        } else if (state == -1) {
            if (action == STAY) {
                return {{-3, 0.5}, {-4, 0.5}};
            } else {
                return {{-3, 0.2}, {-4, 0.8}};
            }
        } else if (state == -2) {
            if (action == STAY) {
                return {{-3, 0.5}, {-4, 0.5}};
            } else {
                return {{-3, 0.1}, {-4, 0.9}};
            }
        } else {
            return {{-4, 1}};
        }
    }

    void reset() {
        state = 0;
        checkpoints.clear();
    }
};

} // namespace rats
