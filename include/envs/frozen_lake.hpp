#pragma once

#include <map>
#include "envs/env.hpp"
#include "pybind/pybind.hpp"


namespace rats { 

class frozen_lake : public environment<int, size_t> {
private:
    py::module_ gym;
    py::object python_env;

    int state;
    int over;
    int checkpoint;
    std::map<size_t, int> checkpoints;
public:
    frozen_lake();
    ~frozen_lake() override = default;

    std::string name() const override { return "FrozenLake 4x4"; }

    std::pair<float, float> reward_range() const override { return {0, 1}; }
    size_t num_actions() const override { return 4; }
    std::vector<size_t> possible_actions() const override { return {0, 1, 2, 3}; }
    size_t get_action(size_t i) const override { return i; }
    int current_state() const override { return state; }
    bool is_over() const override { return over; }
    outcome_t<int> play_action(size_t action) override;

    void restore_checkpoint(size_t id) override;
    void make_checkpoint(size_t id) override;

    void reset() override;
};

frozen_lake::frozen_lake()
: gym(py::module_::import("gymnasium"))
{
    using namespace py::literals;
    python_env = gym.attr("make")("FrozenLake-v1", "map_name"_a="4x4", "is_slippery"_a=true);
}

outcome_t<int> frozen_lake::play_action(size_t action) {
    auto [s, r, o, tranctuated, info] = python_env.attr("step")(action).cast<std::tuple<int, float, bool, bool, py::dict>>();
    this->state = s;
    this->over = o;
    float reward = -1;
    float penalty = o && r < 1;
    return {state, reward, penalty, o};
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

} // namespace rats
