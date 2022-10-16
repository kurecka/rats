#pragma once

#include <random>

extern std::random_device rd;
extern std::mt19937 engine;

/**
 * @brief Set the seed of the random number generator
 * 
 * @param seed 
 */
void set_seed(int seed);

/**
 * @brief Get a random integer in [0, max-1]
 * 
 * @param max
 * @return int 
 */
int unif_int(int max);

/**
 * @brief Get a random integer in [min, max-1]
 * 
 * @param min
 * @param max
 * @return int 
 */
int unif_int(int min, int max);

/**
 * @brief Get a random float in [0, 1]
 * 
 * @return float 
 */
float rand_float();

/**
 * @brief Get a random float in [0, max]
 * 
 * @param max
 * @return float 
 */
float rand_float(float max);

/**
 * @brief Get a random float in [min, max]
 * 
 * @param min
 * @param max
 * @return float 
 */
float rand_float(float min, float max);

/**
 * @brief Get a random boolean with distribution Bernoulli(p)
 * 
 * @param p
 * @return bool
 */
bool bernoulli(float p);
