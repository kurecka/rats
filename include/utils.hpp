#pragma once

#include <vector>
#include <tuple>
#include "rand.hpp"

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
