#pragma once

#include <vector>
#include <tuple>

#include "rand.hpp"

namespace world {

template <typename S>
using outcome_t = std::tuple<S, float, float, bool>;

/*************************************************************************
 * ENVIRONTMENT INTERFACE
 *************************************************************************/

/**
 * @brief Abstract environment class
 * 
 */
template <typename S, typename A>
class environment {
public:
    /**
     * @brief Construct a new environment object
     * 
     */
    environment() = default;

    /**
     * @brief Destroy the environment object
     * 
     */
    virtual ~environment() = default;

    /**
     * @brief Get the action space of the environment
     * 
     * @return std::vector<A> 
     */
    virtual std::vector<A> get_action_space() = 0;

    /**
     * @brief Get current state of the environment
     * 
     * @return S 
     */
    virtual S get_state() = 0;

    /**
     * @brief Return boolean indicating if the environment is over
     * 
     * @return bool
     */
    virtual bool is_over() = 0;

    /**
     * @brief Play an action in the environment
     * 
     * Returns the next state, the reward and punishment, and a boolean indicating whether the episode is over
     * @return outcome_t
     */
    virtual outcome_t<S> play_action(A action, int) = 0;

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


class investor_env : public environment<int, int> {
private:
    int initial_wealth;
    int wealth;
    int target;
    int checkpoint;
public:
    enum investor_action {
        RISKY,
        SAFE
    };

    investor_env(int initial_wealth, int target)
    : initial_wealth(initial_wealth)
    , wealth(initial_wealth)
    , target(target)
    {}

    ~investor_env() = default;

    std::vector<int> get_action_space() override;
    int get_state() override;
    bool is_over() override;
    outcome_t<int> play_action(int action, int player = 0) override;
    void make_checkpoint() override;
    void restore_checkpoint() override;
    void reset() override;
};


/*************************************************************************
 * AGENT INTERFACE
 *************************************************************************/

template <typename S, typename A>
class agent {
public:
    /**
     * @brief Construct a new agent object
     * 
     */
    agent() = default;

    /**
     * @brief Destroy the agent object
     * 
     */
    virtual ~agent() = default;

    /**
     * @brief Get the action to play
     * 
     * @param state 
     * @return A 
     */
    virtual A get_action() = 0;

    /**
     * @brief Pass action outcome to the agent
     * 
     * @param outcome A tuple consisting of state, reward, punishment, over
     */
    virtual void pass_outcome(outcome_t<S> outcome) = 0;

    /**
     * @brief Reset the agent
     * 
     */
    virtual void reset() = 0;
};


/**
 * @brief Agent that plays random actions
 * 
 * @tparam S 
 * @tparam A 
 */
template <typename S, typename A>
class random_agent : public agent<S, A> {
private:
    S initial_state;
    S state;
    std::vector<A> action_space;
public:
    random_agent(S initial_state, std::vector<A> action_space) : state(initial_state), action_space(action_space) {}
    ~random_agent() = default;

    A get_action() override {
        return action_space[unif_int(action_space.size())];
    }

    void pass_outcome(outcome_t<S> outcome) override {
        this->state = std::get<0>(outcome);
    }

    void reset() override {
        state = initial_state;
    }
};

} // namespace world