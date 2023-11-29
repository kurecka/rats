#pragma once

#include <map>
#include "envs/env.hpp"
#include "rand.hpp"

namespace rats { 

class ccpomcp_ex2 : public environment<int, size_t> {
private:
    int state;
    int length;

    std::map<size_t, int> checkpoints;
public:

    enum actions {
        TAKE=0, GO=1
    };

    ccpomcp_ex2(int length)
    : length(length)
    {
        reset();
    }

    ~ccpomcp_ex2() override = default;

    std::string name() const override { return "CC-POMCP Ex2"; }

    std::pair<float, float> reward_range() const override { return {0, 5}; }
    size_t num_actions() const override { return 2; }
    std::vector<size_t> possible_actions() const override { return { TAKE, GO }; }
    size_t get_action(size_t i) const override { return i; }
    int current_state() const override { return state; }
    bool is_over() const override { return state < -2; }
    outcome_t<int> play_action(size_t action) override {
        if (state == 0) {
            if (action == TAKE) {
                state = -1;
                return {-1, 0, 0, false};
            } else {
                state++;
                return {state, 0, 0, false};
            }
        } else if (state > 0 && state < length) {
            if (action == TAKE) {
                state = -3;
                return {-3, 0, 1, true};
            } else {
                state++;
                return {state, 0, 0, false};
            }
        } else if (state == length) {
            if (action == TAKE) {
                state = -2;
                return {-2, 0, 0, false};
            } else {
                state = -3;
                return {-3, 0, 1, true};
            }
        } else if (state == -1) {
            if (action == TAKE) {
                if (rng::unif_float() < 0.5) {
                    state = -3;
                    return {-3, 0.1, 0, true};
                } else {
                    state = -4;
                    return {-4, 0.1, 1, true};
                }
            } else {
                if (rng::unif_float() < 0.2) {
                    state = -3;
                    return {-3, 10, 0, true};
                } else {
                    state = -4;
                    return {-4, 10, 1, true};
                }
            }
        } else if (state == -2) {
            if (action == TAKE) {
                state = -3;
                return {-3, 0, 0, true};
            } else {
                if (rng::unif_float() < 0.1) {
                    state = -3;
                    return {-3, 20, 0, true};
                } else {
                    state = -4;
                    return {-4, 20, 1, true};
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
            if (action == TAKE) {
                return {{-1, 1}};
            } else {
                return {{state+1, 1}};
            }
        } else if (state == length) {
            if (action == TAKE) {
                return {{-1, 1}};
            } else {
                return {{-2, 1}};
            }
        } else if (state == -1) {
            if (action == TAKE) {
                return {{-3, 0.5}, {-4, 0.5}};
            } else {
                return {{-3, 0.2}, {-4, 0.8}};
            }
        } else if (state == -2) {
            if (action == TAKE) {
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
