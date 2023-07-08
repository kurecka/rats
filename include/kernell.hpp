#pragma once

#include <memory>

#include "envs/env.hpp"
#include "agents/agent.hpp"


template <typename S, typename A>
class orchestrator {
private:
    std::unique_ptr<gym::environment<S, A>> env;
    std::unique_ptr<gym::agent<S, A>> agent;

public:
    void load_environment(std::unique_ptr<gym::environment<S, A>> env);
    void load_environment(gym::environment<S, A>* env);
    void load_agent(std::unique_ptr<gym::agent<S, A>> agent);
    void load_agent(gym::agent<S, A>* agent);
    gym::environment_handler<S, A> get_handler() const;

    std::pair<float, float> episode();
    void run(int num_episodes, int num_train_episodes);
};

template <typename S, typename A>
void orchestrator<S, A>::load_environment(std::unique_ptr<gym::environment<S, A>> _env) {
    spdlog::info("Load environment: " + _env->name());
    env = std::move(_env);
    if (agent) {
        agent->set_handler(*env);
    }
}

template <typename S, typename A>
void orchestrator<S, A>::load_environment(gym::environment<S, A>* _env) {
    load_environment(std::unique_ptr<gym::environment<S, A>>(_env));
}

template <typename S, typename A>
gym::environment_handler<S, A> orchestrator<S, A>::get_handler() const {
    return gym::environment_handler<S, A>(*env);
}

template <typename S, typename A>
void orchestrator<S, A>::load_agent(std::unique_ptr<gym::agent<S, A>> _agent) {
    spdlog::info("Load agent: " +_agent->name());
    agent = std::move(_agent);
    if (env) {
        agent->set_handler(*env);
    }
}

template <typename S, typename A>
void orchestrator<S, A>::load_agent(gym::agent<S, A>* _agent) {
    this->load_agent(std::unique_ptr<gym::agent<S, A>>(_agent));
}

template <typename S, typename A>
std::pair<float, float> orchestrator<S, A>::episode() {
    const gym::environment_handler<S, A>& handler = agent->get_handler();
    spdlog::debug("Run episode");
    env->reset();
    spdlog::debug("Environment prepared");
    agent->reset();
    spdlog::debug("Agent prepared");

    while (!env->is_over()) {
        spdlog::debug("Step {}: state={}", handler.get_num_steps(), handler.get_current_state());
        agent->play();
    }

    spdlog::debug("Episode stats:");
    spdlog::debug("  Length: " + std::to_string(handler.get_num_steps()));
    spdlog::debug("  Reward: " + std::to_string(handler.get_reward()));
    spdlog::debug("  Penalty: " + std::to_string(handler.get_penalty()));

    return {handler.get_reward(), handler.get_penalty()};
}

template <typename S, typename A>
void orchestrator<S, A>::run(int num_episodes, int num_train_episodes) {
    spdlog::info("Started");
    spdlog::info("  Agent: " + agent->name());
    spdlog::info("  Environment: " + env->name());
    if (agent->is_trainable()) {
        spdlog::info("Training phase");
        for (int i = 0; i < num_train_episodes; i++) {
            episode();
            agent->train();
        }
    }

    spdlog::info("Evaluation phase");
    float mean_reward = 0;
    float mean_penalty = 0;
    int num_successes = 0;
    for (int i = 0; i < num_episodes; i++) {
        auto [r, p] = episode();
        mean_penalty += p / num_episodes;
        if (p < 0.1f) {
            num_successes++;
            mean_reward += (r - mean_reward) / num_successes;
        }
    }

    spdlog::info("Evaluation results:");
    spdlog::info("  Mean reward: " + std::to_string(mean_reward));
    spdlog::info("  Mean penalty: " + std::to_string(mean_penalty));
}
