#pragma once

#include "world.hpp"
#include <memory>

template <typename S, typename A>
class async_orchestrator {
private:
    std::unique_ptr<world::environment<S, A>> env;
    std::unique_ptr<world::agent<S, A>> agent;

public:
    void load_environment(std::unique_ptr<world::environment<S, A>> env);
    void load_environment(std::unique_ptr<world::environment<S, A>>&& env);
    void load_agent(std::unique_ptr<world::agent<S, A>> agent);
    void load_agent(std::unique_ptr<world::agent<S, A>>&& agent);

    void episode();
    void run(int num_episodes, int num_train_episodes);
};

template <typename S, typename A>
void async_orchestrator<S, A>::load_environment(std::unique_ptr<world::environment<S, A>> env) {
    this->env = std::move(env);
}

template <typename S, typename A>
void async_orchestrator<S, A>::load_environment(std::unique_ptr<world::environment<S, A>>&& env) {
    this->env = std::move(env);
}

template <typename S, typename A>
void async_orchestrator<S, A>::load_agent(std::unique_ptr<world::agent<S, A>> agent) {
    this->agent = std::move(agent);
}

template <typename S, typename A>
void async_orchestrator<S, A>::load_agent(std::unique_ptr<world::agent<S, A>>&& agent) {
    this->agent = std::move(agent);
}

template <typename S, typename A>
void async_orchestrator<S, A>::episode() {
    env->reset();
    agent->reset();
    while (!env->is_over()) {
        env->make_checkpoint();
        auto action = agent->get_action();
        env->restore_checkpoint();
        auto outcome = env->play_action(action);
        agent->pass_outcome(outcome);
    }
}

template <typename S, typename A>
void async_orchestrator<S, A>::run(int num_episodes, int num_train_episodes) {
    if (agent->is_trainable()) {
        for (int i = 0; i < num_train_episodes; i++) {
            episode();
            agent->train();
        }
    }

    for (int i = 0; i < num_episodes; i++) {
        episode();
    }
}