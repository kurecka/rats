#include <random>
#include <vector>
#include <iostream>
#include "rand.hpp"

std::random_device rd;
std::mt19937 engine(rd());
std::uniform_real_distribution<> uni_dist(0, 1);
std::uniform_int_distribution<int> int_uni_dist(0, 1000000000);

void set_seed(int seed) {
    engine.seed(seed);
}

int unif_int(int max) {
    return int_uni_dist(engine) % max;
}

int unif_int(int min, int max) {
    return min + int_uni_dist(engine) % (max - min);
}

float rand_float() {
    return uni_dist(engine);
}

float rand_float(float max) {
    return uni_dist(engine) * max;
}

float rand_float(float min, float max) {
    return min + uni_dist(engine) * (max - min);
}

bool bernoulli(float p) {
    return uni_dist(engine) < p;
}