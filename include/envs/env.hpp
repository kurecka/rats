#pragma once

#include <vector>
#include <tuple>
#include <memory>
#include <map>

#include "rand.hpp"
#include <spdlog/spdlog.h>


namespace rats {

template <typename S>
using outcome_t = std::tuple<S, float, float, bool>; // state, reward, penalty, is_over

/*************************************************************************
 * ENVIRONTMENT INTERFACE
 *************************************************************************/

/**
 * @brief Abstract environment class
 * 
 * TODO: add
 * - templated multireward
 */
template <typename S, typename A>
class environment {
public:
    virtual ~environment() = default;

    virtual std::string name() const = 0;

    /**
     * @brief Get the number of possible actions
     */
    virtual size_t num_actions() const = 0;

    /**
     * @brief Get the vector of all possible actions
     */
    virtual std::vector<A> possible_actions() const = 0;

    /**
     * @brief Get the ith action
     * 
     * @param i index of the action
     */
    virtual A get_action(size_t i) const = 0;

    /**
     * @brief Get the bounds on the reward
     * 
     * @return std::pair<float, float> minimal and maximal reward
     */
    virtual std::pair<float, float> reward_range() const = 0;

    /**
     * @brief Get current state of the environment
     * 
     * @return S 
     */
    virtual S current_state() const = 0;

    /**
     * @brief Return boolean indicating if the environment is over
     * 
     * @return bool
     */
    virtual bool is_over() const = 0;

    /**
     * @brief Play an action in the environment
     * 
     * Returns the next state, the reward and penalty, and a boolean indicating whether the episode is over
     * @return outcome_t
     */
    virtual outcome_t<S> play_action(A action) = 0;

    /**
     * @brief Make a checkpoint of the environment
     * 
     */
    virtual void make_checkpoint(size_t id) = 0;

    /**
     * @brief Restore the environment to the last checkpoint
     * 
     */
    virtual void restore_checkpoint(size_t id) = 0;

    /**
     * @brief Return the action's succesor states' probabilities
     * 
     */
    virtual std::map<S, float> outcome_probabilities(S state, A action) const = 0;

    /**
     * @brief Reset the environment
     * 
     */
    virtual void reset() = 0;
};


/*************************************************************************
 * ENVIRONMENT HANDLER
 *************************************************************************/
template<typename S, typename A>
class environment_handler {
private:
    environment<S, A>* env;
    bool is_simulating = false;
    std::map<size_t, int> checkpoint_num_steps;

    float reward;
    float penalty;
    int num_steps;
    int max_num_steps = 100;
public:
    // environment_handler(environment<S, A>* env) : env(env) {}
    // environment_handler() = default;
    environment_handler(environment<S, A>& _env, int max_num_steps_ = 100)
    : env(&_env)
    , max_num_steps(max_num_steps_)
    {}
    // environment_handler(const environment_handler&, max_num_steps_ = 100) = default;

    ~environment_handler() = default;

    environment_handler& operator=(const environment_handler&) = default;

    float get_reward() const {
        return reward;
    }

    float get_penalty() const {
        return penalty;
    }

    int get_num_steps() const {
        return num_steps;
    }

    void reset() {
        spdlog::debug("Resetting handler");
        reward = penalty = num_steps = 0;
    }

    /**
     * @brief Play an action in the environment
     * 
     * Returns the next state, the reward and penalty, and a boolean indicating whether the episode is over
     * @return outcome_t
     */
    outcome_t<S> play_action(A action) {
        end_sim();
        spdlog::debug("Steps: {} {}", num_steps, max_num_steps);
        if (is_over()) {
            return {get_current_state(), 0, 0, true};
        }
        outcome_t<S> o = env->play_action(action);
        float r = std::get<1>(o);
        float p = std::get<2>(o);
        ++num_steps;
        reward += r;
        penalty += p;

        return o;
    }

    void make_checkpoint(size_t id) {
        env->make_checkpoint(id);
        checkpoint_num_steps[id] = num_steps;
    }

    void restore_checkpoint(size_t id) {
        env->restore_checkpoint(id);
        num_steps = checkpoint_num_steps[id];
    }

    /**
     * @brief Simulate an action in the environment
     * 
     * Returns the next state, the reward and penalty, and a boolean indicating whether the episode is over
     * @return outcome_t
     */
    outcome_t<S> sim_action(A action) {
        if (!is_simulating) {
            make_checkpoint(0);
            is_simulating = true;
        }
        if (is_over()) {
            return {get_current_state(), 0, 0, true};
        }
        ++num_steps;
        return env->play_action(action);
    }

    /**
     * @brief Restore the environment to the last checkpoint
     * 
     */
    void end_sim() {
        if (is_simulating) {
            restore_checkpoint(0);
            is_simulating = false;
        }
    }

    /**
     * @brief Get the bounds on the reward
     * 
     * @return std::pair<float, float> minimal and maximal reward
     */
    std::pair<float, float> reward_range() const {
        return env->reward_range();
    }

    /**
     * @brief Get the number of possible actions
     * 
     * @return size_t
     */
    size_t num_actions() const {
        return env->num_actions();
    }

    /**
     * @brief Get the vector of all possible actions
     * 
     * @return std::vector<A>
     */
    std::vector<A> possible_actions() const {
        return env->possible_actions();
    }

    /**
     * @brief Get the ith action
     * 
     * @param i index of the action
     * @return A
     */
    A get_action(size_t i) const {
        return env->get_action(i);
    }

    /**
     * @brief Get the current state of the environment
     * 
     * @return S 
     */
    S get_current_state() const {
        return env->current_state();
    }

    std::map<S, float> outcome_probabilities(S state, A action) const {
        return env->outcome_probabilities(state, action);
    }

    operator bool() const
    {
        return env;
    }

    bool is_over() const {
        return env->is_over() || num_steps >= max_num_steps;
    }
};
} // namespace rats
