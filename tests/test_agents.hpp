#pragma once
#include "unittest.hpp"
#include <vector>
#include "agents.hpp"


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


UTest(agents, h1_deterministic) {
    // define tree
    using namespace rats::ts;
    using A = int;
    using S = int;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;

    data_t data = {0.1, 5.0, 0.99, 1.0, mockup_handler(), predictor_manager<S, A>()};

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
    AreClose(solver.update_threshold(0.0, 2, 1), 0.0f);

    int a1 = solver.get_action(&root, 0.1);
    ExpectEQ(a1, 2);
    AreClose(solver.update_threshold(0.1, 2, 1), 0.1f);

    int a2 = solver.get_action(&root, 0.5);
    ExpectEQ(a2, 1);
    AreClose(solver.update_threshold(0.5, 1, 1), 0.5f);

    int a3 = solver.get_action(&root, 1.0);
    ExpectEQ(a3, 1);
    AreClose(solver.update_threshold(1.0, 1, 1), 1.0f);

    solver.get_action(&root, 0.3);
    AreClose(solver.update_threshold(0.3, 1, 1), 0.5f);
    AreClose(solver.update_threshold(0.3, 2, 1), 0.1f);
}

UTest(agents, h1_stochastic) {
    // define tree
    using namespace rats::ts;
    using A = int;
    using S = int;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;

    data_t data = {0.1, 5.0, 0.99, 1.0, mockup_handler(), predictor_manager<S, A>()};

    state_node_t root;
    root.state = 0;
    root.common_data = &data;

    expand_state(&root);
    full_expand_action(&(root.children[0]), 1, 0.3, 0.4, 1);
    full_expand_action(&(root.children[1]), 1, 1, 1, 1);
    full_expand_action(&(root.children[1]), 2, 0, 0, 1);
    full_expand_action(&(root.children[2]), 1, 1, 1, 1);
    for (int i = 0; i < 9; ++i) {
        full_expand_action(&(root.children[2]), 2, 0, 0, 1);
    }

    lp::tree_lp_solver<state_node_t> solver;
    int a0 = solver.get_action(&root, 0.0);
    ExpectEQ(a0, 2);
    AreClose(solver.update_threshold(0.0, 2, 1), 0.0f);

    int a1 = solver.get_action(&root, 0.1);
    ExpectEQ(a1, 2);
    AreClose(solver.update_threshold(0.1, 2, 1), 1.f);
    AreClose(solver.update_threshold(0.1, 2, 2), 0.f);

    int a2 = solver.get_action(&root, 0.5);
    ExpectEQ(a2, 1);
    AreClose(solver.update_threshold(0.5, 1, 1), 1.f);
    AreClose(solver.update_threshold(0.5, 1, 2), 0.f);

    int a3 = solver.get_action(&root, 1.0);
    ExpectEQ(a3, 1);
    AreClose(solver.update_threshold(1.0, 1, 1), 1.0f);
    AreClose(solver.update_threshold(1.0, 1, 2), 1.0f);

    solver.get_action(&root, 0.3);
    AreClose(solver.update_threshold(0.3, 1, 1), 1.f);
    AreClose(solver.update_threshold(0.3, 1, 2), 0.f);
    AreClose(solver.update_threshold(0.3, 2, 1), 1.f);
    AreClose(solver.update_threshold(0.3, 2, 2), 0.f);
}

void register_agents_tests() {
    RegisterTest(agents, h1_deterministic)
    RegisterTest(agents, h1_stochastic)
}