#pragma once
#include "unittest.hpp"
#include <vector>
#include "agents.hpp"

#define NTIMES(N) for(int i=0; i<N; ++i)


class mockup_handler {
    public:
        std::vector<int> possible_actions() {
            return {0, 1, 2};
        }
};


struct data_t {
    float risk_thd;
    float exploration_constant;
    float gamma;
    float gammap;
    mockup_handler handler;
    rats::ts::predictor_manager<int, int> predictor;
};


struct sym_node {
    int state;
    int prob_rate;
    std::map<std::pair<int, int>, sym_node> children;
    float observerd_reward=0;
    float observed_penalty=0;
    float rollout_reward=0;
    float rollout_penalty=0;
    bool terminal=false;

    sym_node& operator[](std::pair<int, int> key) {
        return children[key];
    }

    template<typename SN>
    void apply(SN* state_node) {
        state_node->state = state;
        state_node->children.clear();
        state_node->rollout_reward = rollout_reward;
        state_node->rollout_penalty = rollout_penalty;

        for (auto& [key, child] : children) {
            auto& [action, next_state] = key;
            if (action >= state_node->children.size()) state_node->children.resize(action+1);
            if (action >= state_node->actions.size()) {
                state_node->actions.resize(action+1);
                for (int i=0; i<=action; ++i) state_node->actions[i] = i;
            }
            state_node->children[action].parent = state_node;
            state_node->children[action].action = action;
            state_node->children[action].common_data = state_node->common_data;
            for (int i=0; i<child.prob_rate; ++i) {
                full_expand_action(
                    &(state_node->children[action]),
                    next_state,
                    child.observerd_reward,
                    child.observed_penalty,
                    child.terminal
                );
                auto preds = state_node->common_data->predictor.predict_probs(state_node->state, action);
            }
            child.apply(state_node->children[action].children[next_state].get());
        }
    }
};


UTest(agents, h1_deterministic) {
    // define tree
    using namespace rats::ts;
    using A = int;
    using S = int;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;

    data_t data = {0.1, 5.0, 1-1e-3, 1.0, mockup_handler(), predictor_manager<S, A>()};

    state_node_t root;
    root.state = 0;
    root.common_data = &data;

    expand_state(&root);
    full_expand_action(&(root.children[0]), 1, 0.3, 0.4, 1);
    full_expand_action(&(root.children[1]), 1, 0.5, 0.5, 1);
    full_expand_action(&(root.children[2]), 1, 0.1, 0.1, 1);

    lp::tree_lp_solver<state_node_t> solver;
    int a0 = solver.get_action(&root, 0.0);
    ExpectEQ(a0, 2);
    AreCloseEps(solver.update_threshold(0.0, 2, 1), 0.0f, 1e-6);

    int a1 = solver.get_action(&root, 0.1);
    ExpectEQ(a1, 2);
    AreCloseEps(solver.update_threshold(0.1, 2, 1), 0.1f, 1e-6);

    int a2 = solver.get_action(&root, 0.5);
    ExpectEQ(a2, 1);
    AreCloseEps(solver.update_threshold(0.5, 1, 1), 0.5f, 1e-6);

    int a3 = solver.get_action(&root, 1.0);
    ExpectEQ(a3, 1);
    AreCloseEps(solver.update_threshold(1.0, 1, 1), 1.0f, 1e-6);

    solver.get_action(&root, 0.3);
    AreCloseEps(solver.update_threshold(0.3, 1, 1), 0.5f, 1e-6);
    AreCloseEps(solver.update_threshold(0.3, 2, 1), 0.1f, 1e-6);
}

UTest(agents, h1_stochastic) {
    // define tree
    using namespace rats::ts;
    using A = int;
    using S = int;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;

    data_t common_data = {0.1, 5.0, 1-1e-3, 1.0, mockup_handler(), predictor_manager<S, A>()};

    state_node_t root;
    root.state = 0;
    root.common_data = &common_data;

    expand_state(&root);
    // args: &(root.children[action]), state_outcome, reward, probability, terminal
    NTIMES(1) full_expand_action(&(root.children[0]), 1, 0.3, 0.4, 1);

    NTIMES(1) full_expand_action(&(root.children[1]), 1, 1, 1, 1);
    NTIMES(1) full_expand_action(&(root.children[1]), 2, 0, 0, 1);

    NTIMES(1) full_expand_action(&(root.children[2]), 1, 1, 1, 1);
    NTIMES(9) full_expand_action(&(root.children[2]), 2, 0, 0, 1);

    // test outcomes
    // spdlog::set_level(spdlog::level::trace);
    lp::tree_lp_solver<state_node_t> solver;
    int a0 = solver.get_action(&root, 0.0);
    ExpectEQ(a0, 2);
    AreCloseEps(solver.update_threshold(0.0, 2, 1), 0.0f, 1e-6);

    int a1 = solver.get_action(&root, 0.1);
    ExpectEQ(a1, 2);
    AreCloseEps(solver.update_threshold(0.1, 2, 1), 1.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.1, 2, 2), 0.f, 1e-6);

    int a2 = solver.get_action(&root, 0.5);
    ExpectEQ(a2, 1);
    AreCloseEps(solver.update_threshold(0.5, 1, 1), 1.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.5, 1, 2), 0.f, 1e-6);

    int a3 = solver.get_action(&root, 1.0);
    ExpectEQ(a3, 1);
    AreCloseEps(solver.update_threshold(1.0, 1, 1), 1.0f, 1e-6);
    AreCloseEps(solver.update_threshold(1.0, 1, 2), 1.0f, 1e-6);

    solver.get_action(&root, 0.3);
    AreCloseEps(solver.update_threshold(0.3, 1, 1), 1.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.3, 1, 2), 0.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.3, 2, 1), 1.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.3, 2, 2), 0.f, 1e-6);
}

UTest(agents, h2_stochastic) {
    using namespace rats::ts;
    using A = int;
    using S = int;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;

    /*
    s0  ->  a0  ->  s1  ->  a0  ->  s2 r=0.3, p=0.4
        ->  a1  ->  s3:1->  a0  ->  s2 r=1, p=1
                ->  s4:1->  a0  ->  s2 r=0, p=1
                        ->  a1  ->  s2 r=0, p=0
        ->  a2  ->  s5:9->  a0  ->  s2 r=0, p=0
                ->  s6:1->  a0  ->  s2 r=0, p=1
                        ->  a1  ->  s2 r=1, p=1
    
    */

//    spdlog::set_level(spdlog::level::trace);

    sym_node s0 = {0, 1};
    s0[{0, 1}] = {1, 1};
    s0[{0, 1}][{0, 2}] = {2, 1, {}, 0.3, 0.4};
    s0[{1, 3}] = {3, 1};
    s0[{1, 3}][{0, 2}] = {2, 1, {}, 1, 1};
    s0[{1, 4}] = {4, 1};
    s0[{1, 4}][{0, 2}] = {2, 1, {}, 0, 1};
    s0[{1, 4}][{1, 2}] = {2, 1, {}, 0, 0};
    s0[{2, 5}] = {5, 9};
    s0[{2, 5}][{0, 2}] = {2, 1, {}, 0, 0};
    s0[{2, 6}] = {6, 1};
    s0[{2, 6}][{0, 2}] = {2, 1, {}, 0, 1};
    s0[{2, 6}][{1, 2}] = {2, 1, {}, 1, 1};

    data_t common_data = {0.1, 5.0, 1-1e-3, 1.0, mockup_handler(), predictor_manager<S, A>()};
    state_node_t root;
    root.common_data = &common_data;
    s0.apply(&root);

    // std::cout << to_graphviz_tree(root, 6) << std::endl;

    lp::tree_lp_solver<state_node_t> solver;
    int a0 = solver.get_action(&root, 0.0);
    ExpectEQ(a0, 2);
    AreCloseEps(solver.update_threshold(0.0, 2, 5), 0.0f, 1e-6);
    AreCloseEps(solver.update_threshold(0.0, 2, 6), 0.0f, 1e-6);

    int a1 = solver.get_action(&root, 0.1);
    ExpectEQ(a1, 2);
    AreCloseEps(solver.update_threshold(0.1, 2, 5), 0.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.1, 2, 6), 1.f, 1e-6);

    int a2 = solver.get_action(&root, 0.5);
    ExpectEQ(a2, 1);
    AreCloseEps(solver.update_threshold(0.5, 1, 3), 1.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.5, 1, 4), 0.f, 1e-6);

    int a3 = solver.get_action(&root, 1.0);
    ExpectEQ(a3, 1);
    AreCloseEps(solver.update_threshold(1.0, 1, 3), 1.0f, 1e-6);
    AreCloseEps(solver.update_threshold(1.0, 1, 4), 1.0f, 1e-6);

    solver.get_action(&root, 0.3);
    AreCloseEps(solver.update_threshold(0.3, 1, 3), 1.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.3, 1, 4), 0.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.3, 2, 5), 0.f, 1e-6);
    AreCloseEps(solver.update_threshold(0.3, 2, 6), 1.f, 1e-6);
}


void register_agents_tests() {
    RegisterTest(agents, h1_deterministic)
    RegisterTest(agents, h1_stochastic)
    RegisterTest(agents, h2_stochastic)
}