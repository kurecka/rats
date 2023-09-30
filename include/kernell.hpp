#pragma once

#include <memory>

#include "envs/env.hpp"
#include "agents/agent.hpp"
#include "string_utils.hpp"

namespace rats {

template <typename S, typename A>
class orchestrator {
private:
    std::shared_ptr<environment<S, A>> env;
    std::shared_ptr<agent<S, A>> agnt;

public:
    void load_environment(std::shared_ptr<environment<S, A>> env);
    void load_environment(environment<S, A>* env);
    void load_agent(std::shared_ptr<agent<S, A>> agnt);
    void load_agent(agent<S, A>* agnt);
    environment_handler<S, A> get_handler() const;

    std::pair<float, float> episode();
    void run(int num_episodes, int num_train_episodes);
};

template <typename S, typename A>
void orchestrator<S, A>::load_environment(std::shared_ptr<environment<S, A>> _env) {
    spdlog::info("Load environment: " + _env->name());
    env = _env;
    if (agnt) {
        agnt->set_handler(*env);
    }
}

template <typename S, typename A>
void orchestrator<S, A>::load_environment(environment<S, A>* _env) {
    spdlog::info("Load environment: " + _env->name());
    load_environment(std::shared_ptr<environment<S, A>>(_env));
}

template <typename S, typename A>
environment_handler<S, A> orchestrator<S, A>::get_handler() const {
    return environment_handler<S, A>(*env);
}

template <typename S, typename A>
void orchestrator<S, A>::load_agent(std::shared_ptr<agent<S, A>> _agnt) {
    spdlog::info("Load agent: " +_agnt->name());
    agnt = _agnt;
    if (env) {
        agnt->set_handler(*env);
    }
}

template <typename S, typename A>
void orchestrator<S, A>::load_agent(agent<S, A>* _agnt) {
    spdlog::info("Load agent: " +_agnt->name());
    this->load_agent(std::shared_ptr<agent<S, A>>(_agnt));
}

template <typename S, typename A>
std::pair<float, float> orchestrator<S, A>::episode() {
    const environment_handler<S, A>& handler = agnt->get_handler();
    spdlog::debug("Run episode");
    env->reset();
    spdlog::debug("Environment prepared");
    agnt->reset();
    spdlog::debug("Agent prepared");

    while (!env->is_over()) {
        spdlog::trace("Step {}: state={}", handler.get_num_steps(), handler.get_current_state());
        agnt->play();
    }

    spdlog::debug("Episode stats:");
    spdlog::debug("  Length: " + to_string(handler.get_num_steps()));
    spdlog::debug("  Reward: " + to_string(handler.get_reward()));
    spdlog::debug("  Penalty: " + to_string(handler.get_penalty()));

    return {handler.get_reward(), handler.get_penalty()};
}

template <typename S, typename A>
void orchestrator<S, A>::run(int num_episodes, int num_train_episodes) {
    spdlog::info("Started");
    spdlog::info("  Agent: " + agnt->name());
    spdlog::info("  Environment: " + env->name());
    if (agnt->is_trainable()) {
        spdlog::info("Training phase");
        for (int i = 0; i < num_train_episodes; i++) {
            episode();
            agnt->train();
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
    spdlog::info("  Mean reward: " + to_string(mean_reward));
    spdlog::info("  Mean penalty: " + to_string(mean_penalty));
}

} // namespace rats