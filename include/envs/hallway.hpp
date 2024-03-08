#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cstdint>

#include "envs/env.hpp"
#include "rand.hpp"


namespace rats { 

struct map_manager {
    enum tile {
        EMPTY = -1,
        WALL = -2,
        TRAP = -3,
        GOLD = -4
    };
    enum hallway_action {
        LEFT = 0,
        DOWN = 1,
        RIGHT = 2,
        UP = 3
    };
    std::vector<int> data;
    int width;
    int height;
    int start;
    int num_gold;
    uint64_t initial_gold_mask;
    
    /**
     * @brief Parse a string into a map
     * 
     * Example map:
     * ######
     * #.B..#
     * #.#..#
     * #.T.G#
     * #GT..#
     * ######
    */
    map_manager(std::string str)
    : width(0), height(0)
    {
        std::stringstream ss(str);
        std::string line;
        num_gold = 0;
        initial_gold_mask = 0;
        while (std::getline(ss, line)) {
            if (line.size() == 0) continue;
            if (width == 0) {
                width = line.size();
            } else if (width != line.size()) {
                throw std::runtime_error("Invalid map: lines have different sizes:\n");
            }
            for (char c : line) {
                switch (c) {
                    case '#':
                        data.push_back(WALL);
                        break;
                    case '.':
                        data.push_back(EMPTY);
                        break;
                    case 'T':
                        data.push_back(TRAP);
                        break;
                    case 'G':
                        data.push_back(num_gold++);
                        if (num_gold > 64) {
                            throw std::runtime_error("Invalid map: too many golds (max 64)");
                        }
                        initial_gold_mask <<= 1;
                        initial_gold_mask |= 1;
                        break;
                    case 'B':
                        data.push_back(EMPTY);
                        start = data.size() - 1;
                        break;
                    default:
                        throw std::runtime_error("Invalid map: unknown character");
                }
            }

            height++;
        }

        spdlog::debug("Loading map:");
        spdlog::debug(" Map size: {}x{}", width, height);
        spdlog::debug(" Start position: {}", start);
        spdlog::debug(" Number of golds: {}", num_gold);
        spdlog::debug(" Mask: {}", initial_gold_mask);
        spdlog::debug(" Map:\n{}", str);
    }

    // position, gold_mask, tile, hit wall
    std::tuple<int, uint64_t, int, bool> move(size_t action, int pos, uint64_t gold_mask) const {

        int x = pos % width;
        int y = pos / width;
        int new_x = x;        int new_y = y;
        switch (action) {
            case LEFT:
                new_x = x > 0 ? x - 1 : x;
                break;
            case RIGHT:
                new_x = x < width - 1 ? x + 1 : x;
                break;
            case UP:
                new_y = y > 0 ? y - 1 : y;
                break;
            case DOWN:
                new_y = y < height - 1 ? y + 1 : y;
                break;
        }
        int new_pos = new_y * width + new_x;
        int new_gold_mask = gold_mask;
        bool hit = false;
        if (data[new_pos] == WALL) {
            new_pos = pos;
            hit = true;
        }
        if (data[new_pos] >= 0) {
            uint64_t all = -1;
            new_gold_mask &= all ^ (1 << data[new_pos]);
        }
        return {new_pos, new_gold_mask, data[new_pos], hit};
    }
};

class hallway : public environment<std::pair<int, uint64_t>, size_t> {
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
    hallway(std::string, float trap_prob = 0.2f, float slide_prob=0.f);

    std::string name() const override { return "Hallway"; }

    std::pair<float, float> reward_range() const override { return {0, 1}; }
    size_t num_actions() const override { return 4; }
    std::vector<action_t> possible_actions(state_t = {}) const override { return {0, 1, 2, 3}; }
    size_t get_action(size_t i) const override { return i; }
    state_t current_state() const override { return {position, gold_mask}; }
    bool is_over() const override { return over; }
    outcome_t<state_t> play_action(size_t action) override;

    void restore_checkpoint(size_t id) override;
    void make_checkpoint(size_t id) override;

    std::map<state_t, float> outcome_probabilities(state_t state, action_t action) const override;

    void reset() override;
};

hallway::hallway(std::string map_str, float trap_prob, float slide_prob)
: m(map_str), trap_prob(trap_prob), slide_prob(slide_prob)
{
    reset();
}

outcome_t<typename hallway::state_t> hallway::play_action(size_t action) {
    if (over) {
        throw std::runtime_error("Cannot play action: environment is over");
    }
    auto [new_pos, new_gold_mask, tile, hit] = m.move(action, position, gold_mask);
    if (rng::unif_float() < slide_prob && new_pos != position) {
        size_t slide_action = (action + 3 + 2 * rng::unif_int(2)) % 4;
        std::tie(new_pos, new_gold_mask, tile, hit) = m.move(slide_action, new_pos, gold_mask);
    }    

    float reward = new_gold_mask != gold_mask;
    if (hit) reward -= 0.00001f;
    float penalty = (tile == map_manager::TRAP) && (rng::unif_float() < trap_prob);
    if (penalty > 0) { new_pos = -1; }
    over = (new_gold_mask == 0) || penalty > 0;
    position = new_pos;
    gold_mask = new_gold_mask;
    return {{position, gold_mask}, reward, penalty, over};
}

void hallway::make_checkpoint(size_t id) {
    if (id == 0) {
        checkpoint = {position, gold_mask};
    } else {
        checkpoints[id] = {position, gold_mask};
    }
}

void hallway::restore_checkpoint(size_t id) {
    if (id == 0) {
        std::tie(position, gold_mask) = checkpoint;
    } else {
        std::tie(position, gold_mask) = checkpoints[id];
    }
    over = position == -1;
}

void hallway::reset() {
    position = m.start;
    gold_mask = m.initial_gold_mask;
    over = false;
}

std::map<typename hallway::state_t, float> hallway::outcome_probabilities(typename hallway::state_t s, size_t a) const {
    auto [pos, gold_mask] = s;
    auto [new_pos, new_gold_mask, tile, hit] = m.move(a, pos, gold_mask);
    std::map<typename hallway::state_t, float> outcomes;

    size_t slide_action_1 = (a + 3) % 4;
    size_t slide_action_2 = (a + 5) % 4;
    auto [new_pos1, new_gold_mask1, tile1, hit1] = m.move(slide_action_1, new_pos, gold_mask);
    auto [new_pos2, new_gold_mask2, tile2, hit2] = m.move(slide_action_2, new_pos, gold_mask);

    float non_slide_prob = 1 - slide_prob + (slide_prob / 2) * (hit1 + hit2);
    if (tile == map_manager::TRAP) {
        outcomes.insert({{{-1, new_gold_mask}, non_slide_prob * trap_prob}, {{new_pos, new_gold_mask}, non_slide_prob * (1 - trap_prob)}});
    } else {
        outcomes.insert({{{new_pos, new_gold_mask}, non_slide_prob}});
    }

    float slided_prob = slide_prob / 2;
    if (!hit1) {
        if (tile1 == map_manager::TRAP) {
            outcomes.insert({{{-1, new_gold_mask1}, slided_prob * trap_prob}, {{new_pos1, new_gold_mask1}, slided_prob * (1 - trap_prob)}});
        } else {
            outcomes.insert({{{new_pos1, new_gold_mask1}, slided_prob}});
        }
    }
    if (!hit2) {
        if (tile2 == map_manager::TRAP) {
            outcomes.insert({{{-1, new_gold_mask2}, slided_prob * trap_prob}, {{new_pos2, new_gold_mask2}, slided_prob * (1 - trap_prob)}});
        } else {
            outcomes.insert({{{new_pos2, new_gold_mask2}, slided_prob}});
        }
    }

    return outcomes;
}

float hallway::solve_exactly() const {

}

} // namespace rats
