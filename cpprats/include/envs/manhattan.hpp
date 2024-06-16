#pragma once

#include <iostream>
#include <map>
#include <string>
#include "envs/env.hpp"
#include "pybind/pybind.hpp"


namespace rats { 

/*
 *  manhattan environment class, largely just forwards calls to
 *  manhattan/manhattan.py 
 *
 *  currently implemented - just a simulator, i.e. states are positions on the
 *  map, can use play_action, possible_actions, etc. to interact
 *
 *  not implemented - methods used for LP, i.e. getting reward, terminal
 *  state and reward_range()
 *
 *  to adjust implementation, see rats/manhattan/manhattan.py
 */
class manhattan : public environment<std::tuple<std::string, std::map< std::string, float >, bool>, int> {

private:
    py::object python_env;
    using state_t = std::tuple<std::string, std::map< std::string, float >, bool>;

public:
    ~manhattan() override = default;

    manhattan(  float capacity, 
                const std::vector<std::string> &targets,
                const std::map<std::string, float> &periods,
                std::string init_state,
                float cons_thd);

    std::string name() const override;

    std::pair<float, float> reward_range() const override;
    size_t num_actions() const override;
    std::vector<int> possible_actions(state_t state) const override; 
    int get_action(size_t i) const override;
    state_t current_state() const override;
    bool is_over() const override;
    bool is_terminal( state_t ) const override;

    std::pair<float, float> get_expected_reward( state_t state, int, state_t succ ) const override;
    outcome_t<state_t> play_action(int action) override;

    void restore_checkpoint(size_t id) override;
    void make_checkpoint(size_t id) override;
    std::map<state_t, float> outcome_probabilities(state_t state, int action) const override;

    void animate_simulation(int interval=100, const std::string &filename = "map.html");
    void reset() override;
};

manhattan::manhattan( float capacity, 
                      const std::vector<std::string> &targets,
                      const std::map<std::string, float> &periods,
                      std::string init_state="",
                      float cons_thd=10.0f)
{
    using namespace py::literals;
    python_env = py::module_::import("manhattan").attr("ManhattanEnv")(capacity, targets, periods, init_state, cons_thd);
}

outcome_t<manhattan::state_t> manhattan::play_action(int action) {
    auto [ state, r, p, over ] = python_env.attr("play_action")(action).cast< std::tuple< state_t, float, float, bool > >();


    return { state, r, p, over };
}

std::string manhattan::name() const {
    return python_env.attr("name")().cast<std::string>();
}

// TODO: not supported 
std::pair< float, float > manhattan::reward_range() const {
    return {0, 0};
}

size_t manhattan::num_actions() const{
    return python_env.attr("num_actions")().cast<size_t>();
}

int manhattan::get_action(size_t id) const {
    return python_env.attr("get_action")(id).cast<int>();
}

std::vector< int > manhattan::possible_actions( manhattan::state_t s ) const {
    return python_env.attr("possible_actions")(s).cast<std::vector< int > >();
}

// TODO: not supported 
std::pair< float, float > manhattan::get_expected_reward( manhattan::state_t s, int a, manhattan::state_t s2 ) const {
    return {0, 0};
}

bool manhattan::is_terminal( manhattan::state_t ) const {
    return false;
}

void manhattan::make_checkpoint(size_t id) {
    python_env.attr("make_checkpoint")(id);
}

void manhattan::restore_checkpoint(size_t id) {
    python_env.attr("restore_checkpoint")(id);
}

void manhattan::reset() {
    python_env.attr("reset")();
}

manhattan::state_t manhattan::current_state() const {
    return python_env.attr("current_state")().cast<manhattan::state_t>();
}

bool manhattan::is_over() const {
    return python_env.attr("is_over")().cast<bool>();
}

std::map<manhattan::state_t, float> manhattan::outcome_probabilities(manhattan::state_t s, int action) const {
    return python_env.attr("outcome_probabilities")(s, action).cast<std::map<manhattan::state_t, float>>();
}

void manhattan::animate_simulation( int interval, const std::string& filename) {
    python_env.attr("animate_simulation")(interval, filename);
}

} // namespace rats
