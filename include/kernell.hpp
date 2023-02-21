#pragma once

#include <memory>

#include "world.hpp"


template <typename S>
class orchestrator {
private:
    std::unique_ptr<world::environment<S>> env;
    std::unique_ptr<world::agent<S>> agent;

public:
    void load_environment(std::unique_ptr<world::environment<S>> env);
    void load_environment(world::environment<S>* env);
    void load_agent(std::unique_ptr<world::agent<S>> agent);
    void load_agent(world::agent<S>* agent);

    void episode();
    void run(int num_episodes, int num_train_episodes);
};

template <typename S>
void orchestrator<S>::load_environment(std::unique_ptr<world::environment<S>> env) {
    logger.info("Load environment: " + env->name());
    this->env = std::move(env);
    if (agent) {
        agent->set_handler(*(this->env));
    }
}

template <typename S>
void orchestrator<S>::load_environment(world::environment<S>* env) {
    load_environment(std::unique_ptr<world::environment<S>>(env));
}

template <typename S>
void orchestrator<S>::load_agent(std::unique_ptr<world::agent<S>> agent) {
    logger.info("Load agent: " + agent->name());
    this->agent = std::move(agent);
    if (env) {
        this->agent->set_handler(*env);
    }
}

template <typename S>
void orchestrator<S>::load_agent(world::agent<S>* agent) {
    this->load_agent(std::unique_ptr<world::agent<S>>(agent));
}

template <typename S>
void orchestrator<S>::episode() {
    logger.info("Run episode");
    env->reset();
    agent->reset();

    int l = 0;

    while (!env->is_over()) {
        l++;
        agent->play();
    }

    logger.info("Episode stats:");
    logger.info("  Length: " + std::to_string(l));
    logger.info("  Final value: " + std::to_string(env->current_state()));
}

template <typename S>
void orchestrator<S>::run(int num_episodes, int num_train_episodes) {
    logger.info("Started");
    logger.info("  Agent: " + agent->name());
    logger.info("  Environmentx: " + env->name());
    if (agent->is_trainable()) {
        logger.info("Training phase");
        for (int i = 0; i < num_train_episodes; i++) {
            episode();
            agent->train();
        }
    }

    logger.info("Evaluation phase");

    for (int i = 0; i < num_episodes; i++) {
        episode();
    }
}