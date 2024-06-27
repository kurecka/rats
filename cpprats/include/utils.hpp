#pragma once

#include <spdlog/spdlog.h>
#include <vector>
#include <tuple>
#include "rand.hpp"


std::pair<float, float> penalties2probs(float penalty1, float penalty2, float thd);

struct mixture {
    size_t vals[2];
    float probs[2];
    float penalties[2];

    int last_sampled_index;

    mixture() = default;

    /**
     * @brief A stochastic mixture of two outcomes with associated penalties
     * 
     * @param min_outcome Outcome with minimum penalty
     * @param max_outcome Outcome with maximum penalty
     * @param min_penalty Penalty associated with min_outcome
     * @param max_penalty Penalty associated with max_outcome
     * @param thd Threshold for the mixture
     * 
     * Represents a mixture of two outcomes with associated penalties. The mixture is such that the expected penalty
     * is equal to the threshold thd (if possible).
     * 
     * - If the maximum penalty is less than or equal to the threshold, the mixture is deterministic and the action with
     *  the maximum penalty is always chosen.
     * - If the minimum penalty is greater than or equal to the threshold, the mixture is deterministic and the action with
     *  the minimum penalty is always chosen.
     * - Otherwise, the mixture is probabilistic and the probabilities are computed such that the expected penalty is
     *  equal to the threshold.
     */
    mixture(size_t min_outcome, size_t max_outcome, float min_penalty, float max_penalty, float thd)
    {
        if (max_penalty <= thd) {
            vals[0] = vals[1] = max_outcome;
            penalties[0] = penalties[1] = max_penalty;
            probs[0] = 0;
            probs[1] = 1;
        }
        else if (min_penalty >= thd) {
            vals[1] = vals[0] = min_outcome;
            penalties[1] = penalties[0] = min_penalty;
            probs[0] = 1;
            probs[1] = 0;
        }
        else {
            vals[0] = min_outcome;
            vals[1] = max_outcome;
            penalties[0] = min_penalty;
            penalties[1] = max_penalty;
            std::tie(probs[0], probs[1]) = penalties2probs(min_penalty, max_penalty, thd);
        }
    }

    size_t sample() {
        last_sampled_index = rng::unif_float(0, 1) < probs[0] ? 0 : 1;
        return vals[last_sampled_index];
    }

    float update_thd(float thd, float immediate_penalty = 0, float gammap = 1.0f) {
        float new_thd = 0;
        if (vals[0] == vals[1]) {
            new_thd = (thd - immediate_penalty) / gammap;
        } else {
            float prob_action = probs[last_sampled_index];
            double alt_thd = penalties[1 - last_sampled_index];
            new_thd = (thd - prob_action * immediate_penalty - (1 - prob_action) * alt_thd) / (prob_action * gammap);
        }
        return std::max(new_thd, 0.0f);
    }

    template<typename T>
    float expectation(T v1, T v2) const {
        return probs[0] * v1 + probs[1] * v2;
    }

    float last_penalty() const {
        return penalties[last_sampled_index];
    }
};


template <typename T>
struct mixture_legacy {
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


mixture_legacy<size_t> greedy_mix(const std::vector<float>& rs, const std::vector<float>& ps, float thd);
