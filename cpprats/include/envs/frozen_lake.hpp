#pragma once

#include <map>
#include "envs/env.hpp"
#include "pybind/pybind.hpp"


namespace rats { 

class frozen_lake : public environment<int, size_t> {
    enum frozen_lake_action {
        LEFT = 0,
        DOWN = 1,
        RIGHT = 2,
        UP = 3
    };
private:
    py::module_ gym;
    py::object python_env;

    int state;
    int over;
    int checkpoint;
    int env_size;
    std::map<size_t, int> checkpoints;
public:
    frozen_lake();
    ~frozen_lake() override = default;

    std::string name() const override { return "FrozenLake 4x4"; }

    std::pair<float, float> reward_range() const override { return {0, 1}; }
    size_t num_actions() const override { return 4; }
    std::vector<size_t> possible_actions(int = {}) const override { return {0, 1, 2, 3}; }
    size_t get_action(size_t i) const override { return i; }
    int current_state() const override { return state; }
    bool is_over() const override { return over; }
    bool is_terminal( int ) const override;

    // TODO: not supported for now, perhaps a way of getting it from gym somehow
    std::pair<float, float> get_expected_reward( int state, size_t, int succ ) const override { 
        return {0, 0};
    }

    outcome_t<int> play_action(size_t action) override;

    void restore_checkpoint(size_t id) override;
    void make_checkpoint(size_t id) override;

    std::map<int, float> outcome_probabilities(int state, size_t action) const override;

    void reset() override;
};

frozen_lake::frozen_lake()
: gym(py::module_::import("gymnasium"))
{
    using namespace py::literals;
    python_env = gym.attr("make")("FrozenLake-v1", "map_name"_a="4x4", "is_slippery"_a=true);
    env_size = python_env.attr("observation_space").cast<int>();
}

outcome_t<int> frozen_lake::play_action(size_t action) {
    auto [s, r, o, tranctuated, info] = python_env.attr("step")(action).cast<std::tuple<int, float, bool, bool, py::dict>>();
    this->state = s;
    this->over = o;
    float reward = -1;
    float penalty = o && r < 1;
    return {state, reward, penalty, o};
}

bool frozen_lake::is_terminal( int state ) const {
    // terminal state is only the last square
    return state == env_size - 1;
}


void frozen_lake::make_checkpoint(size_t id) {
    if (id == 0) {
        checkpoint = state;
    } else {
        checkpoints[id] = state;
    }
}

void frozen_lake::restore_checkpoint(size_t id) {
    if (id == 0) {
        state = checkpoint;
    } else {
        state = checkpoints[id];
    }
    python_env.attr("s") = state;
}

void frozen_lake::reset() {
    auto [s, info] = python_env.attr("reset")().cast<std::tuple<int, py::dict>>();
    this->state = s;
    this->over = false;
}

std::map<int, float> frozen_lake::outcome_probabilities(int s, size_t action) const {
    std::map<int, float> probs;
    if (action != LEFT) {
        int dest = s % 4 != 3 ? s + 1 : s;
        probs[dest] += 1/3.0f;
    }
    if (action != RIGHT) {
        int dest = s % 4 != 0 ? s - 1 : s;
        probs[dest] += 1/3.0f;
    }
    if (action != DOWN) {
        int dest = s > 3 ? s - 4 : s;
        probs[dest] += 1/3.0f;
    }
    if (action != UP) {
        int dest = s < 12 ? s + 4 : s;
        probs[dest] += 1/3.0f;
    }
    return probs;
}

} // namespace rats
