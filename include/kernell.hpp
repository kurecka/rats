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

    std::pair<float, float> episode();
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
std::pair<float, float> orchestrator<S>::episode() {
    const world::environment_handler<S>& handler = agent->get_handler();
    logger.debug("Run episode");
    env->reset();
    agent->reset();

    while (!env->is_over()) {
        agent->play();
    }

    logger.debug("Episode stats:");
    logger.debug("  Length: " + std::to_string(handler.get_num_steps()));
    logger.debug("  Reward: " + std::to_string(handler.get_reward()));
    logger.debug("  Penalty: " + std::to_string(handler.get_penalty()));

    return {handler.get_reward(), handler.get_penalty()};
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
    float mean_reward = 0;
    float mean_penalty = 0;
    for (int i = 0; i < num_episodes; i++) {
        auto [r, p] = episode();
        mean_reward += r / num_episodes;
        mean_penalty += p / num_episodes;
    }

    logger.info("Evaluation results:");
    logger.info("  Mean reward: " + std::to_string(mean_reward));
    logger.info("  Mean penalty: " + std::to_string(mean_penalty));
}