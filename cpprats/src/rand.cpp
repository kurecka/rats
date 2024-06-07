#include <random>
#include <vector>
#include <iostream>
#include "rand.hpp"

std::random_device rng::rd = std::random_device();
std::mt19937 rng::engine = std::mt19937(rng::rd());

std::uniform_int_distribution<int> rng::int_uni_dist;
std::uniform_real_distribution<float> rng::f_uni_dist;
std::uniform_real_distribution<double> rng::d_uni_dist;
