#pragma once

#include <random>

class rng {
private:
    static unsigned int _seed;
    static std::random_device rd;
    static std::mt19937 engine;

    static std::uniform_int_distribution<int> int_uni_dist;
    static std::uniform_real_distribution<float> f_uni_dist;
    static std::uniform_real_distribution<double> d_uni_dist;


    public:
    static void init(unsigned int seed = 0) {
      _seed = seed;
      engine.seed(_seed);
    }

    /**
     * @brief Set the seed of the random number generator
     * 
     * @param seed seed for the random number generator
     */
    static void set_seed(unsigned int seed) {
        engine.seed(seed);
    }

    /**
     * @brief Get a random number from the standard normal distribution
    */
    static float normal(float mean = 0, float dev = 0.1) {
        std::normal_distribution<float> dist(mean, dev);
        return dist(engine);
    }

    /**
     * @brief Get a random integer in [0, max-1]
     * 
     * @param max upper bound on the random number
     * @return int 
     */
    template <typename T>
    static T unif_int(T max) {
        return static_cast<T>(int_uni_dist(engine)) % max;
    }

    /**
     * @brief Get a random integer in [min, max-1]
     * 
     * @param min lower bound on the random number (included)
     * @param max upper bound on the random number (excluded)
     * @return int 
     */
    template <typename T>
    static T unif_int(T min, T max) {
        return static_cast<T>(int_uni_dist(engine)) % (max - min) + min;
    }

    /**
     * @brief Get a random float in [0, 1]
     * 
     * @return float 
     */
    static float unif_float() {
        return f_uni_dist(engine);
    }

    /**
     * @brief Get a random float in [0, max]
     * 
     * @param max upper bound on the random number
     * @return float 
     */
    static float unif_float(float max) {
        return f_uni_dist(engine) * max;
    }

    /**
     * @brief Get a random float in [min, max]
     * 
     * @param min lower bound on the random number
     * @param max upper bound on the random number
     * @return float 
     */
    static float unif_float(float min, float max) {
        return min + f_uni_dist(engine) * (max - min);
    }

    /**
     * @brief Get a random double in [0, 1]
     * 
     * @return double 
     */
    static double unif_double() {
        return d_uni_dist(engine);
    }

    /**
     * @brief Get a random double in [0, max]
     * 
     * @param max upper bound on the random number
     * @return double 
     */
    static double unif_double(double max) {
        return d_uni_dist(engine) * max;
    }

    /**
     * @brief Get a random double in [min, max]
     * 
     * @param min lower bound on the random number
     * @param max upper bound on the random number
     * @return double 
     */
    static double unif_double(double min, double max) {
        return min + d_uni_dist(engine) * (max - min);
    }

    /**
     * @brief Get a random boolean with distribution Bernoulli(p)
     * 
     * @param p probability of getting true
     * @return bool
     */
    template <typename T>
    static bool bernoulli(T p) {
        return d_uni_dist(engine) < p;
    }

    /**
     * @brief Get a random sample from discrete distribution
     * 
     * @param distribution container with probabilities (summing to 1)
     * @return sample from the discrete distribution
     */
    template <typename T>
    static int custom_discrete(T distribution) {
        std::discrete_distribution<> d (distribution.begin(), distribution.end());
        return d(engine);
    }
};
