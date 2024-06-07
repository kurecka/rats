#pragma once

#include <map>
#include "envs/env.hpp"
#include "rand.hpp"

namespace rats { 

class ccpomcp_ex1 : public environment<int, size_t> {
private:
    int state;
    std::map<size_t, int> checkpoints;
public:

    enum states {
        S0, S2, S3, S7, S8, S9, star
    };

    enum actions {
        A1=0, A4=1, A5=2, A6=3
    };

    ccpomcp_ex1()
    {
        reset();
    }

    ~ccpomcp_ex1() override = default;

    std::string name() const override { return "CC-POMCP Ex1"; }

    std::pair<float, float> reward_range() const override { return {0, 1}; }
    size_t num_actions() const override { return 4; }
    std::vector<size_t> possible_actions(int = {}) const override { return {
        A1, A4, A5, A6
    }; }
    size_t get_action(size_t i) const override { return i; }
    int current_state() const override { return state; }
    bool is_over() const override { return state == star; }

    bool is_terminal( int state ) const override { return state == star;  }

    // TODO : not supported for now
    std::pair< float, float > get_expected_reward( int state , size_t action, int succ ) const override {
        return {0, 0};
    }

    outcome_t<int> play_action(size_t action) override {
        if (state == S0) {
            if (action == A1) {
                if (rng::unif_float() < 0.5) {
                    state = S2;
                    return {state, 0, 0, false};
                } else {
                    state = S3;
                    return {state, 0, 0, false};
                }
            } else {
                state = star;
                return {state, 0, 10, true};
            }
        } else if (state == S2) {
            if (action == A4) {
                state = S7;
                return {state, 1, 1, false};
            } else if (action == A5){
                state = S8;
                return {state, 0, 0, false};
            } else {
                state = star;
                return {state, 0, 10, true};
            }
        } else if (state == S3) {
            if (action == A6) {
                state = S9;
                return {S9, 0, 1, false};
            } else {
                state = star;
                return {state, 0, 10, true};
            }
        } else {
            state = star;
            return {star, 0, 0, true};
        }
    }

    void restore_checkpoint(size_t id) override {
        state = checkpoints[id];
    }
    void make_checkpoint(size_t id) override {
        checkpoints[id] = state;
    }
    std::map<int, float> outcome_probabilities(int state, size_t action) const override {
        if (state == S0) {
            if (action == A1) {
                return {{S2, 0.5}, {S3, 0.5}};
            } else {
                return {{star, 1}};
            }
        } else if (state == S2) {
            if (action == A4) {
                return {{S7, 1}};
            } else if (action == A5){
                return {{S8, 1}};
            } else {
                return {{star, 1}};
            }
        } else if (state == S3) {
            if (action == A6) {
                return {{S9, 1}};
            } else {
                return {{star, 1}};
            }
        } else {
            return {{star, 1}};
        }
    }

    void reset() {
        state = S0;
        checkpoints.clear();
    }
};

} // namespace rats
