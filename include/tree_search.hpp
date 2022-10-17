#pragma once

#include "world.hpp"

#include <vector>
#include <map>

namespace world {
namespace st {

struct action_data {
    int num_visits;
    float average_payoff;
    float average_penalty;
    float prob_prediction;
};

template <typename S, typename A>
struct node {
    A incoming_action;
    S state;
    node *parent = nullptr;
    int num_visits = 0;
    float payoff_prediction = 0;
    float penalty_prediction = 0;
    float max_v = 0;
    float min_v = 0;
    float reward = 0;
    float penalty = 0;

    std::map<std::pair<A, S>, node> children;
    std::map<A, action_data> a_data;

    node(
        node* parent, A incoming_action, S state,
        float payoff_prediction, float penalty_prediction,
        const std::vector<float>& prob_prediction,
        const std::vector<A>& action_space)
    : state(state)
    , incoming_action(incoming_action)
    , parent(parent)
    , payoff_prediction(payoff_prediction)
    , penalty_prediction(penalty_prediction)
    {
        for (int i = 0; i < action_space.size(); i++) {
            a_data[action_space[i]] = {0, 0, prob_prediction[i]};
        }
    }

    void expand(
        const std::vector<std::pair<A, S>>& as_pairs,
        const std::vector<float>& action_space,
        const std::vector<std::tuple<float, float, std::vector<float>>>& predictions
    ) {
        for (int i = 0; i < as_pairs.size(); i++) {
            auto [a, s] = as_pairs[i];
            auto [payoff, penalty, prob] = predictions[i];
            children[as_pairs[i]] = node(this, a, s, payoff, penalty, std::move(prob), action_space);
        }
    }

    bool is_leaf() const {
        return children.empty();
    }

    bool is_root() const {
        return parent == nullptr;
    }
};

template <typename S, typename A>
class search_tree : public agent<S, A> {
private:
    int max_depth;
    int num_sim;
    float gamma;

    node<S, A> root;
    environment<S, A> *env;
    A last_action;
    std::vector<node<S, A>> history;

    void simulate();
    virtual const A& select_action(node<S, A> *nod) = 0;
    virtual const A& explore_action(node<S, A> *nod) = 0;
    void expand(node<S, A> *nod);
    void backprop(node<S, A> *nod);
    std::tuple<float, float, std::vector<float>> eval_history(node<S, A>* nod, const A& action, const S& state) = 0;

    void prune(const S& state);
public:
    /* Agent interface */
    search_tree(environment<S, A>* env);
    const A& get_action() override;
    void pass_outcome(outcome_t<S> outcome) override;
    void reset() override;
};

template <typename S, typename A>
void search_tree<S, A>::simulate() {
    env->restore_checkpoint();

    int depth = 0;
    node<S, A> *current = &root;
    
    float reward;
    float penalty;
    while (!current->is_leaf()) {
        A action = explore_action(current);
        auto& [s, g, p, e] = env->play_action(action);
        reward = g;
        penalty = p;
        current = &current->children[{action, s}];
        depth++;
    }

    current->reward = reward;
    current->penalty = penalty;

    if (depth < max_depth && !env->is_over()) {
        expand(current);
    }

    backprop(current);
}

template <typename S, typename A>
void search_tree<S, A>::backprop(node<S, A> *nod) {
    ++nod->num_visits;

    float val = nod->payoff_prediction;
    float reg = nod->penalty_prediction;

    while (!nod->is_root()) {
        val = nod->reward + gamma * val;
        reg = nod->penalty + gamma * reg;
        const A& a = nod->incoming_action;
        const S& s = nod->state;
        node<S, A>* parent = nod->parent;
        ++(parent->num_visits);
        ++(parent->a_data[a].num_visits);
        parent->a_data[a].average_payoff += (val - parent->a_data[a].average_payoff) / parent->a_data[a].num_visits;
        parent->a_data[a].average_penalty += (reg - parent->a_data[a].average_penalty) / parent->a_data[a].num_visits;

        nod = parent;
    }
}

template <typename S, typename A>
void search_tree<S, A>::expand(node<S, A> *nod) {
    auto [s, g, p, e] = env->get_state();
    auto action_space = env->get_action_space();
    std::vector<std::pair<A, S>> as_pairs;
    std::vector<std::tuple<float, float, std::vector<float>>> predictions;
    for (auto a : action_space) {
        std::vector<S> next_states = env->get_next_states(nod->state, a);
        for (auto s : next_states) {
            as_pairs.push_back({a, s});
            predictions.push_back(predict(nod, a, s));
        }
    }
    nod->expand(as_pairs, action_space, env->get_action_space(), predictions);
}

template <typename S, typename A>
void search_tree<S, A>::prune(const S& state) {
    history.push_back(std::move(root));
    root = std::move(history.back().children[{last_action, state}]);
    root.parent = nullptr;
    history.back.children.clear();
}

/************************ AGENT INTERFACE ************************/
template <typename S, typename A>
const A& search_tree<S, A>::get_action() {
    history.clear();

    for (int i = 0; i < num_sim; i++) {
        simulate();
    }
    last_action = select_action();
}

template <typename S, typename A>
void search_tree<S, A>::pass_outcome(outcome_t<S> outcome) {
    prune(std::get<0>(outcome));
}

template <typename S, typename A>
void search_tree<S, A>::reset() {
    auto [payoff, penalty, prob] = eval_history(nullptr, A(), env->get_state());
    root = node<S, A>(nullptr, A(), env->get_state(), payoff, penalty, prob, env->get_action_space());
}

template <typename S, typename A>
search_tree<S, A>::search_tree(environment<S, A>* env) : env(env) {
    reset();
}

} // namespace st
} // namespace world