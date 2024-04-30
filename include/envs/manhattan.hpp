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
 *  TODO: should prob adjust states to pos + energy + positions of orders for learning
 *  later, is_terminal() will then simply return whether a given state has <= 0 energy
 *
 *  to adjust implementation, see manhattan/manhattan.py
 */
class manhattan : public environment<std::string, size_t> {

private:
    py::object python_env;
    using state_t = std::string;

public:
    ~manhattan() override = default;

    manhattan( float capacity, 
               const std::vector<std::string> &targets,
               const std::vector<std::string> &reloads,
               std::string init_state);

    std::string name() const override;

    std::pair<float, float> reward_range() const override;
    size_t num_actions() const override;
    std::vector<size_t> possible_actions(state_t state) const override; 
    size_t get_action(size_t i) const override;
    state_t current_state() const override;
    bool is_over() const override;
    bool is_terminal( state_t ) const override;

    std::pair<float, float> get_expected_reward( state_t state, size_t, state_t succ ) const override;
    outcome_t<state_t> play_action(size_t action) override;

    void restore_checkpoint(size_t id) override;
    void make_checkpoint(size_t id) override;
    std::map<std::string, float> outcome_probabilities(state_t state, size_t action) const override;

    void reset() override;
};

manhattan::manhattan( float capacity, 
                      const std::vector<std::string> &targets,
                      const std::vector<std::string> &reloads,
                      std::string init_state="")
{
    using namespace py::literals;
    python_env = py::module_::import("manhattan.manhattan").attr("ManhattanEnv")(capacity, targets, reloads, init_state);
}

outcome_t<std::string> manhattan::play_action(size_t action) {
    auto [ state, r, p, over ] = python_env.attr("play_action")(action).cast< std::tuple< std::string, float, float, bool > >();


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

size_t manhattan::get_action(size_t id) const {
    return python_env.attr("get_action")(id).cast<size_t>();
}

// TODO: arg not supported, just returns from current state
std::vector< size_t > manhattan::possible_actions( std::string s ) const {
    return python_env.attr("possible_actions")().cast<std::vector< size_t > >();
}

// TODO: not supported 
std::pair< float, float > manhattan::get_expected_reward( std::string s, size_t a, std::string s2 ) const {
    return {0, 0};
}

bool manhattan::is_terminal( std::string ) const {
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
    return python_env.attr("current_state").cast<manhattan::state_t>();
}

bool manhattan::is_over() const {
    return python_env.attr("is_over")().cast<bool>();
}

std::map<std::string, float> manhattan::outcome_probabilities(std::string s, size_t action) const {
    return python_env.attr("outcome_possibilities")(action).cast<std::map<std::string, float>>();
}

} // namespace rats
