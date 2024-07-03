#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cstdint>

#include "envs/env.hpp"
#include "envs/hallway.hpp"
#include "rand.hpp"


namespace rats { 

class continuing_hallway : public environment<std::pair<int, uint64_t>, size_t> {
public:
    using action_t = size_t;
    using state_t = std::pair<int, uint64_t>;
private:
    int position;
    uint64_t gold_mask;
    bool over;
    state_t checkpoint;
    std::map<size_t, state_t> checkpoints;
    map_manager m;
    float trap_prob;
    float slide_prob;
public:
    continuing_hallway(std::string, float trap_prob = 0.2f, float slide_prob=0.f);

    std::string name() const override { return "ContHallway"; }
    std::string to_string(state_t s) const { return fmt::format("({}, {}; {})", s.first % m.width, s.first / m.width, s.second); }
    int get_width() const { return m.width; }

    std::pair<float, float> get_expected_reward( state_t, action_t, state_t ) const override;
    std::pair<float, float> penalty_range() const override { return {0, trap_prob}; }
    std::pair<float, float> reward_range() const override { return {0, 1}; }
    size_t num_actions() const override { return 4; }
    std::vector<action_t> possible_actions(state_t = {}) const override { return {0, 1, 2, 3}; }
    size_t get_action(size_t i) const override { return i; }
    state_t current_state() const override { return {position, gold_mask}; }
    bool is_over() const override { return over; }
    bool is_terminal( state_t s ) const override;

    outcome_t<state_t> play_action(size_t action) override;

    void restore_checkpoint(size_t id) override;
    void make_checkpoint(size_t id) override;

    std::map<state_t, float> outcome_probabilities(state_t state, action_t action) const override;

    void reset() override;
};

continuing_hallway::continuing_hallway(std::string map_str, float trap_prob, float slide_prob)
: m(map_str), trap_prob(trap_prob), slide_prob(slide_prob)
{
    reset();
}

outcome_t<typename continuing_hallway::state_t> continuing_hallway::play_action(size_t action) {
    if (over) {
        throw std::runtime_error("Cannot play action: environment is over");
    }
    auto [new_pos, new_gold_mask, tile, hit] = m.move(action, position, gold_mask);

    // roll for slides
    if (rng::unif_float() < slide_prob && new_pos != position) {
        size_t slide_action = (action + 3 + 2 * rng::unif_int(2)) % 4;
        std::tie(new_pos, new_gold_mask, tile, hit) = m.move(slide_action, new_pos, gold_mask);
    }    

    auto [ reward, penalty ] = get_expected_reward( {position, gold_mask}, action, {new_pos, new_gold_mask} );

    over = new_gold_mask == 0;
    position = new_pos;
    gold_mask = new_gold_mask;

    return {{position, gold_mask}, reward, penalty, over};
}


std::pair<float, float> continuing_hallway::get_expected_reward( state_t state, action_t action, state_t succ ) const { 
    auto [old_pos, old_mask] = state;
    auto [new_pos, new_mask] = succ;

    // find out if hit a wall using move method
    auto [ _newpos, _newmask, _tile, hit] = m.move(action, old_pos, old_mask);

    float reward = new_mask != old_mask;
    float penalty = 0;

    if (hit) reward -= 0.00001f;
    if (_tile == map_manager::TRAP) penalty = trap_prob;

    return {reward, penalty};
}

void continuing_hallway::make_checkpoint(size_t id) {
    if (id == 0) {
        checkpoint = {position, gold_mask};
    } else {
        checkpoints[id] = {position, gold_mask};
    }
}

void continuing_hallway::restore_checkpoint(size_t id) {
    if (id == 0) {
        std::tie(position, gold_mask) = checkpoint;
    } else {
        std::tie(position, gold_mask) = checkpoints[id];
    }
    over = gold_mask == 0;
}

void continuing_hallway::reset() {
    position = m.start;
    gold_mask = m.initial_gold_mask;
    over = false;
}

bool continuing_hallway::is_terminal( state_t s ) const {
    auto [ pos, gold_mask ] = s;
    // terminal if died or collected all the gold
    return gold_mask == 0;
}

std::map<typename continuing_hallway::state_t, float> continuing_hallway::outcome_probabilities(typename continuing_hallway::state_t s, size_t a) const {
    auto [pos, gold_mask] = s;
    auto [new_pos, new_gold_mask, tile, hit] = m.move(a, pos, gold_mask);

    std::map<typename continuing_hallway::state_t, float> outcomes;
    size_t slide_action_1 = (a + 3) % 4;
    size_t slide_action_2 = (a + 5) % 4;
    auto [new_pos1, new_gold_mask1, tile1, hit1] = m.move(slide_action_1, new_pos, gold_mask);
    auto [new_pos2, new_gold_mask2, tile2, hit2] = m.move(slide_action_2, new_pos, gold_mask);

    float non_slide_prob = hit ? 1 : 1 - slide_prob + (slide_prob / 2) * (hit1 + hit2);
    outcomes.insert({{{new_pos, new_gold_mask}, non_slide_prob}});

    float slided_prob = slide_prob / 2;
    if (!hit && !hit1) {
        outcomes.insert({{{new_pos1, new_gold_mask1}, slided_prob}});
    }
    if (!hit && !hit2) {
        outcomes.insert({{{new_pos2, new_gold_mask2}, slided_prob}});
    }

    return outcomes;
}

} // namespace rats
