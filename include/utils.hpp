#pragma once

#include <spdlog/spdlog.h>
#include <vector>
#include <tuple>
#include "rand.hpp"


std::pair<float, float> penalties2probs(float penalty1, float penalty2, float thd);


struct action_policy {
    size_t actions[2];
    float probs[2];
    float penalties[2];

    int last_sampled_action;

    action_policy() = default;

    action_policy(size_t min_action, size_t max_action, float min_penalty, float max_penalty, float thd)
    {
        if (max_penalty <= thd) {
            actions[0] = actions[1] = max_action;
            penalties[0] = penalties[1] = max_penalty;
            probs[0] = 0;
            probs[1] = 1;
        }
        else if (min_penalty >= thd) {
            actions[1] = actions[0] = min_action;
            penalties[1] = penalties[0] = min_penalty;
            probs[0] = 1;
            probs[1] = 0;
        }
        else {
            actions[0] = min_action;
            actions[1] = max_action;
            penalties[0] = min_penalty;
            penalties[1] = max_penalty;
            std::tie(probs[0], probs[1]) = penalties2probs(min_penalty, max_penalty, thd);
        }
    }

    size_t sample() {
        last_sampled_action = rng::unif_int(0, 1) < probs[0] ? 0 : 1;
        return actions[last_sampled_action];
    }

    float update_thd(float thd, float immediate_penalty = 0) {
        float new_thd = 0;
        if (actions[0] == actions[1]) {
            new_thd = thd;
        } else {
            int other_action = 1 - last_sampled_action;
            double alt_thd = penalties[other_action];
            new_thd = (thd - probs[last_sampled_action] * immediate_penalty - (1 - probs[last_sampled_action]) * alt_thd) / probs[last_sampled_action];
        }
        return std::clamp(new_thd, 0.0f, 1.0f);
    }
};


template <typename T>
struct mixture {
    T v1, v2;
    float p1, p2;
    bool deterministic = false;

    T operator()() const {
        if (deterministic) {
            return p2 > p1 ? v2 : v1;
        } else {
            return rng::unif_float() < p2 ? v2 : v1;
        }
    }

    template <typename U>
    U operator()(U u1, U u2) {
        if (deterministic) {
            return p2 > p1 ? u2 : u1;
        } else {
            return rng::unif_float() < p2 ? u2 : u1;
        }
    }
};


mixture<size_t> greedy_mix(const std::vector<float>& rs, const std::vector<float>& ps, float thd);
