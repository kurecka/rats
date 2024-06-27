#pragma once

#include <vector>
#include <tuple>
#include <memory>

#include "spdlog/spdlog.h"
#include "rand.hpp"
#include "envs/env.hpp"

namespace rats {

template <typename S>
using outcome_t = std::tuple<S, float, float, bool>; // state, reward, penalty, is_over


/*************************************************************************
 * AGENT INTERFACE
 *************************************************************************/
template <typename S, typename A>
class agent {
protected:
    environment_handler<S, A> handler;
public:
    /**
     * @brief Set the handler object
     * 
     * @param _handler the handler that controls the environment
     */
    void set_handler(environment_handler<S, A> _handler) {
        spdlog::info("Setting agent handler");
        handler = _handler;
    }

    /**
     * @brief Set the handler object
     * 
     * @param _env environment that is xontrolled by the handler
     */
    void set_handler(environment<S, A>& _env) {
        spdlog::info("Setting agent handler");
        handler = environment_handler<S, A>(_env);
        _env.reset();
    }

    const environment_handler<S, A>& get_handler() const {
        return handler;
    }

    /**
     * @brief Construct a new agent object 
     * 
     */
    agent(environment_handler<S, A> _handler)
    : handler(_handler) {}

    agent() {}

    /**
     * @brief Destroy the agent object
     * 
     */
    virtual ~agent() = default;

    /**
     * @brief Reset the agent
     * 
     */
    virtual void reset() {
        handler.reset();
    }

    /**
     * @brief Play through the environment handler
     *
     */
    virtual void play() = 0;

    /**
     * @brief Train the agent after the episode is over
     * 
     */
    virtual void train() {}

    /**
     * @brief Return boolean indicating if the agent is trainable
     * 
     */
    constexpr virtual bool is_trainable() const {
        // TODO: make this into a trait?
        return false;
    }

    /**
     * @brief Get the name of the agent
     * 
     * @return std::string 
     */
    virtual std::string name() const = 0;
};

} // namespace rats
