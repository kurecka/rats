#pragma once

#include <vector>
#include <tuple>
#include <memory>

#include "rand.hpp"
#include "logging.hpp"

namespace world {

template <typename S>
using outcome_t = std::tuple<S, float, float, bool>; // state, reward, penalty, is_over
using action_t = int;

/*************************************************************************
 * ENVIRONTMENT INTERFACE
 *************************************************************************/

/**
 * @brief Abstract environment class
 * 
 */
template <typename S>
class environment {
public:
    virtual std::string name() const = 0;

    /**
     * @brief Get the number of possible actions
     * 
     * @return const int 
     */
    virtual int num_actions() const = 0;

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
    virtual outcome_t<S> play_action(action_t action) = 0;

    /**
     * @brief Make a checkpoint of the environment
     * 
     */
    virtual void make_checkpoint() = 0;

    /**
     * @brief Restore the environment to the last checkpoint
     * 
     */
    virtual void restore_checkpoint() = 0;

    /**
     * @brief Reset the environment
     * 
     */
    virtual void reset() = 0;
};


/*************************************************************************
 * ENVIRONMENT HANDLER
 *************************************************************************/
template<typename S>
class environment_handler {
private:
    environment<S>* env;
    bool is_simulating = false;
public:
    // environment_handler(environment<S>* env) : env(env) {}
    environment_handler() = default;
    environment_handler(environment<S>& env) : env(&env) {}
    environment_handler(const environment_handler&) = default;

    ~environment_handler() = default;

    environment_handler& operator=(const environment_handler&) = default;

    /**
     * @brief Play an action in the environment
     * 
     * Returns the next state, the reward and penalty, and a boolean indicating whether the episode is over
     * @return outcome_t
     */
    outcome_t<S> play_action(action_t action) {
        if (is_simulating) {
            env->restore_checkpoint();
            is_simulating = false;
        }
        return env->play_action(action);
    };

    /**
     * @brief Simulate an action in the environment
     * 
     * Returns the next state, the reward and penalty, and a boolean indicating whether the episode is over
     * @return outcome_t
     */
    outcome_t<S> sim_action(action_t action) {
        if (!is_simulating) {
            env->make_checkpoint();
            is_simulating = true;
        }
        return env->play_action(action);
    };

    /**
     * @brief Get the number of possible actions
     * 
     * @return int 
     */
    int num_actions() const {
        return env->num_actions();
    }

    /**
     * @brief Get current state of the environment
     * 
     * @return S 
     */
    S current_state() const {
        return env->current_state();
    }

    operator bool() const
    {
        return env;
    }
};


/*************************************************************************
 * AGENT INTERFACE
 *************************************************************************/

template <typename S>
class agent {
protected:
    environment_handler<S> handler;
public:
    /**
     * @brief Set the handler object
     * 
     * @param handler 
     */
    void set_handler(environment_handler<S> handler) {
        logger.info("Setting agent handler");
        handler = handler;
    }

    /**
     * @brief Set the handler object
     * 
     * @param env 
     */
    void set_handler(environment<S>& env) {
        logger.info("Setting agent handler");
        handler = environment_handler<S>(env);
        env.reset();
    }

    /**
     * @brief 
     * 
     */
    agent() {};

    /**
     * @brief Destroy the agent object
     * 
     */
    virtual ~agent() = default;

    /**
     * @brief Reset the agent
     * 
     */
    virtual void reset() = 0;

    /**
     * @brief Play throygh the environment handler
     * 
     * @param state 
     * @return A 
     */
    virtual void play() = 0;

    /**
     * @brief Train the agent
     * 
     */
    virtual void train() {};

    /**
     * @brief Return boolean indicating if the agent is trainable
     * 
     */
    virtual bool is_trainable() const {
        return false;
    }

    /**
     * @brief Get the name of the agent
     * 
     * @return std::string 
     */
    virtual std::string name() const = 0;
};

} // namespace world