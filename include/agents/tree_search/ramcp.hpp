#pragma once

#include "tree_search.hpp"
#include "string_utils.hpp"
#include "ortools/linear_solver/linear_solver.h"
#include <string>
#include <vector>

namespace rats {
namespace ts {

using namespace operations_research;

template <typename S, typename A>
struct ramcp_data {
    float risk_thd;
    float exploration_constant;
    environment_handler<S, A>& handler;
};

template<typename S, typename A, typename DATA, typename V>
A select_action_uct(state_node<S, A, DATA, V, point_value>* node, bool explore) {

    float c = node->common_data->exploration_constant;

    auto& children = node->children;
    float max_v = std::max_element(children.begin(), children.end(),
                    [](auto& l, auto& r){ return l.q.first < r.q.first; })->q.first;
    float min_v = std::min_element(children.begin(), children.end(),
                    [](auto& l, auto& r){ return l.q.first < r.q.first; })->q.first;

    size_t idxa = 0;
    float max_uct = 0, uct_value = 0;
    for (size_t i = 0; i < children.size(); ++i) {
        uct_value = ((children[i].q.first - Vmin) / (Vmax - Vmin)) +
            c * static_cast<float>(std::sqrt(std::log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
        );

        if (uct_value > max_uct) {
            max_uct = uct_value;
            idxa = i;
        }
    }

    return node->actions[idxa];
}

/*********************************************************************
 * @brief exact ramcp uct agent
 * 
 * @tparam S State type
 * @tparam A Action type
 *********************************************************************/

template <typename S, typename A>
class ramcp : public agent<S, A> {
    using data_t = ramcp_data<S, A>;
    using v_t = std::pair<float, float>;
    using q_t = std::pair<float, float>;
    using uct_state_t = state_node<S, A, data_t, v_t, q_t>;
    using uct_action_t = action_node<S, A, data_t, v_t, q_t>;
    
    constexpr auto static select_leaf_f = select_leaf<S, A, data_t, v_t, q_t, select_action_uct<S, A, data_t, v_t>, void_descend_callback<S, A, data_t, v_t, q_t>>;
    constexpr auto static propagate_f = propagate<S, A, data_t, v_t, q_t, uct_prop_v_value<S, A, data_t>, uct_prop_q_value<S, A, data_t>>;

private:
    int max_depth;
    int num_sim;
    float risk_thd;
    float gamma;

    data_t common_data;

    std::unique_ptr<uct_state_t> root;
    std::unique_ptr<MPSolver> solver;
public:
    ramcp(
        environment_handler<S, A> _handler,
        int _max_depth, int _num_sim, float _risk_thd, float _gamma,
        float _exploration_constant = 5.0 
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , risk_thd(_risk_thd)
    , gamma(_gamma)
    , common_data({_risk_thd, initial_lambda, _exploration_constant, agent<S, A>::handler})
    , root(std::make_unique<uct_state_t>())
    {
        // Create the linear solvers with the GLOP backend.
        solver = std::make_unique<MPSolver>(MPSolver::CreateSolver("GLOP"));
        reset();
    }

    void play() override {
        spdlog::debug("Play: {}", name());

        for (int i = 0; i < num_sim; i++) {
            spdlog::trace("Simulation {}", i);
            spdlog::trace("Select leaf");
            uct_state_t* leaf = select_leaf_f(root.get(), true, max_depth);
            spdlog::trace("Expand leaf");
            expand_state(leaf);
            spdlog::trace("Rollout");
            rollout(leaf);
            spdlog::trace("Propagate");
            propagate_f(leaf, gamma);
            agent<S, A>::handler.sim_reset();
        }

        solver->Clear();
    }

    std::pair<std::unordered_map<A, MPVariable* const>, std::unordered_map<S, std::vector<MPVariable* const>>> define_LP_policy(
            uct_state_t* root, double risk_thd, MPSolver* solver) {

        std::unordered_map<A, MPVariable* const> policy;

        // root succesor == leaf parent, for alternative risk calculations
        std::unordered_map<S, std::vector<MPVariable* const>> leaves;

        //assert(!root->leaf());

        size_t ctr = 0; //counter

        MPObjective* const objective = solver->MutableObjective();

        MPConstraint* const risk_cons = solver->MakeRowConstraint(0.0, risk_thd); // risk

        MPVariable* const r = solver->MakeIntVar(1, 1, "r"); // (1)

        MPConstraint* const action_sum = solver->MakeRowConstraint(0, 0); // setting 0 0 for equality could cause rounding problems
        action_sum->SetCoefficient(r, -1); // sum of action prob == prob of parent (2)

        auto& actions = root->actions;
        for (auto ac_it = actions.begin(), auto child_it = children.begin();
             ac_it != actions.end(); ++ac_it, ++child_it) {

            MPVariable* const ac = solver_policy->MakeNumVar(0.0, 1.0, std::to_string(ctr++));
            action_sum->SetCoefficient(ac, 1);
            policy.insert({*ac_it, ac}); // add to policy to access solution

            auto states_distr = common_data.handler.outcome_probabilities(root->state, *ac_it);

            for (auto it = child_it->children.begin(); it != child_it->children.end(); ++it) {

                MPVariable* const st = solver_policy->MakeNumVar(0.0, 1.0, std::to_string(ctr++));

                // st = ac * delta (3)
                MPConstraint* const ac_st = solver_policy->MakeRowConstraint(0, 0);
                ac_st->SetCoefficient(st, -1);
                double delta = states_distr[it->first];
                ac_st->SetCoefficient(ac, delta);

                LP_policy_rec(it->second.get(), st, ctr, risk_cons, objective, 1, solver, leaves, it->first);
            }
        }

        objective->SetMaximization();

        return {policy, leaves};
    }

    void LP_policy_rec(uct_state_t* node, MPVariable* const var, size_t& ctr, MPConstraint* const risk_cons,
                        MPObjective* const objective, size_t node_depth, MPSolver* solver,
                        std::unordered_map<S, std::vector<MPVariable* const>> leaves, S root_succ) {

        if (node->leaf()) {

            // objective
            double coef = (node->payoff / std::pow(tree->gamma, tree->root->his.actions.size()))  + std::pow(tree->gamma, node_depth) * node->v;
            objective->SetCoefficient(var, coef);

            // risk
            risk_cons->SetCoefficient(var, node->r);

            // alternative risk
            leaves[root_succ].push_back(var);
            return;
        }

        MPConstraint* const action_sum = solver->MakeRowConstraint(0, 0); // setting 0 0 for equality could cause rounding problems
        action_sum->SetCoefficient(var, -1); // sum of action prob == prob of parent (2)

        auto& actions = node->actions;
        for (auto ac_it = actions.begin(), auto child_it = children.begin();
             ac_it != actions.end(); ++ac_it, ++child_it) {

            MPVariable* const ac = solver_policy->MakeNumVar(0.0, 1.0, std::to_string(ctr++)); // x_h,a
            action_sum->SetCoefficient(ac, 1);

            auto states_distr = common_data.handler.outcome_probabilities(node->state, *ac_it);

            for (auto it = child_it->children.begin(); it != child_it->children.end(); ++it) {

                MPVariable* const st = solver_policy->MakeNumVar(0.0, 1.0, std::to_string(ctr++));

                // st = ac * delta (3)
                MPConstraint* const ac_st = solver_policy->MakeRowConstraint(0, 0);
                ac_st->SetCoefficient(st, -1);
                double delta = states_distr[it->first];
                ac_st->SetCoefficient(ac, delta);

                LP_policy_rec(it->second.get(), st, ctr, risk_cons, objective, node_depth + 1, solver, leaves, it->first);
            }
        }
    }

    void reset() override {
        spdlog::debug("Reset: {}", name());
        agent<S, A>::reset();
        risk_thd = common_data.risk_thd;
        root = std::make_unique<uct_state_t>();
        root->common_data = &common_data;
    }

    std::string name() const override {
        return "ramcp";
    }
};

}
}