#pragma once

#include "tree_search.hpp"
#include "string_utils.hpp"
#include "rand.hpp"
#include "ortools/linear_solver/linear_solver.h"
#include <string>
#include <vector>
#include <map>
// #include <map>

namespace rats {
namespace ts {

using namespace operations_research;

template <typename S, typename A>
struct ramcp_data {
    float risk_thd;
    float exploration_constant;
    float gamma;
    float gammap;
    environment_handler<S, A>& handler;
};


template<typename SN>
struct select_action_uct {
    size_t operator()(SN* node, bool /*explore*/) const {

        float c = node->common_data->exploration_constant;

        auto& children = node->children;
        float max_v = std::max_element(children.begin(), children.end(),
                        [](auto& l, auto& r){ return l.q.first < r.q.first; })->q.first;
        float min_v = std::min_element(children.begin(), children.end(),
                        [](auto& l, auto& r){ return l.q.first < r.q.first; })->q.first;

        size_t idxa = 0;
        float max_uct = 0, uct_value = 0;
        for (size_t i = 0; i < children.size(); ++i) {
            uct_value = ((children[i].q.first - min_v) / (max_v - min_v)) +
                c * static_cast<float>(std::sqrt(std::log(node->num_visits + 1) / (children[i].num_visits + 0.0001))
            );

            if (uct_value > max_uct) {
                max_uct = uct_value;
                idxa = i;
            }
        }

        return idxa;
    }
};

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
    using state_node_t = state_node<S, A, data_t, v_t, q_t>;
    using action_node_t = action_node<S, A, data_t, v_t, q_t>;

    using select_action_t = select_action_uct<state_node_t>;
    using descend_callback_t = void_fn<state_node_t*, A, action_node_t*, S, state_node_t*>;
    constexpr auto static select_action_f = select_action_t();
    constexpr auto static select_leaf_f = select_leaf<state_node_t, select_action_t, descend_callback_t>;
    constexpr auto static propagate_f = propagate<state_node_t, uct_prop_v_value<state_node_t>, uct_prop_q_value<action_node_t>>;

private:
    int max_depth;
    int num_sim;
    int sim_time_limit;
    float risk_thd;

    data_t common_data;

    std::unique_ptr<state_node_t> root;
    std::unique_ptr<MPSolver> solver;
public:
    ramcp(
        environment_handler<S, A> _handler,
        int _max_depth, float _risk_thd, float _gamma,
        int _num_sim = 100, int _sim_time_limit = 0,
        float _exploration_constant = 5.0, float _gammap = 0 
    )
    : agent<S, A>(_handler)
    , max_depth(_max_depth)
    , num_sim(_num_sim)
    , sim_time_limit(_sim_time_limit)
    , risk_thd(_risk_thd)
    , common_data({_risk_thd, _exploration_constant, _gamma, _gammap, agent<S, A>::handler})
    , root(std::make_unique<state_node_t>())
    , solver(std::unique_ptr<MPSolver>(MPSolver::CreateSolver("GLOP")))
    {
        // Create the linear solvers with the GLOP backend.
        root->state = agent<S, A>::handler.get_current_state();
        // spdlog::debug("root {}", to_string(root->state));
        reset();
    }

    void simulate(int i) {
        state_node_t* leaf = select_leaf_f(root.get(), true, max_depth);
        expand_state(leaf);
        rollout(leaf);
        propagate_f(leaf);
        agent<S, A>::handler.end_sim();
    }

    void play() override {
        spdlog::debug("Play: {}", name());

        root->state = common_data.handler.get_current_state();
        // spdlog::debug("root {}", to_string(root->state));

        if (sim_time_limit > 0) {
            auto start = std::chrono::high_resolution_clock::now();
            auto end = start + std::chrono::milliseconds(sim_time_limit);
            int i = 0;
            while (std::chrono::high_resolution_clock::now() < end) {
                simulate(i++);
            }
        } else {
            for (int i = 0; i < num_sim; ++i) {
                simulate(i);
            }
        }

        auto [policy, leaf_risk] = define_LP_policy(risk_thd);
        MPSolver::ResultStatus result_status = solver->Solve();

        if (result_status != MPSolver::OPTIMAL) {

            // if risk_thd is infeasable we relax the bound
            auto risk_obj = define_LP_risk();
            result_status = solver->Solve();
            //assert(result_status == MPSolver::OPTIMAL);

            double alt_thd = risk_obj->Value();
            std::tie(policy, leaf_risk) = define_LP_policy(alt_thd);
            result_status = solver->Solve();
            //assert(result_status == MPSolver::OPTIMAL);
        }

        // sample action
        std::vector<double> policy_distr;
        for (auto it = policy.begin(); it != policy.end(); it++) {
            policy_distr.emplace_back(it->second->solution_value());
            // spdlog::debug("actions prob: {}", policy_distr.back());
        }

        int sample = rng::custom_discrete(policy_distr);

        A a = std::next(std::begin(policy), sample)->first;

        auto [s, r, p, t] = common_data.handler.play_action(a);
        spdlog::debug("Play action: {}", to_string(a));
        spdlog::debug(" Result: s={}, r={}, p={}", to_string(s), r, p);

        action_node_t* an = root->get_child(a);
        if (an->children.find(s) == an->children.end()) {
            expand_action(an, s, r, p, t);
        }

        // adjust risk thd, assuming only observed penalty == leaf in F
        double alt_risk = 0;
        for (auto& [child, leaves] : leaf_risk) {

            if (child == s)
                continue;

            for (auto& leaf : leaves) {

                alt_risk += leaf->solution_value();
            }
        }


        // assert(alt_risk <= risk_thd);

        auto states_distr = common_data.handler.outcome_probabilities(root->state, a);
        risk_thd = (risk_thd - alt_risk) / (policy[a]->solution_value() * states_distr[s]);

        std::unique_ptr<state_node_t> new_root = an->get_child_unique_ptr(s);
        root = std::move(new_root);
        root->get_parent() = nullptr;
    }

    std::pair<std::map<A, MPVariable* const>, std::map<S, std::vector<MPVariable*>>> define_LP_policy(
            double risk_thd) {

        solver->Clear();

        std::map<A, MPVariable* const> policy;

        // root succesor == leaf parent, for alternative risk calculations
        std::map<S, std::vector<MPVariable*>> leaves;

        //assert(!root->leaf());

        size_t ctr = 0; //counter
        double payoff = 0;

        MPObjective* const objective = solver->MutableObjective();

        MPConstraint* const risk_cons = solver->MakeRowConstraint(0.0, risk_thd); // risk
        // spdlog::debug("risk_thd: {}", risk_thd);

        MPVariable* const r = solver->MakeIntVar(1, 1, "r"); // (1)

        MPConstraint* const action_sum = solver->MakeRowConstraint(0, 0); // setting 0 0 for equality could cause rounding problems
        action_sum->SetCoefficient(r, -1); // sum of action prob == prob of parent (2)

        auto& actions = root->actions;
        auto& children = root->children;
        auto child_it = children.begin();
        for (auto ac_it = actions.begin();
             ac_it != actions.end(); ++ac_it, ++child_it) {

            MPVariable* const ac = solver->MakeNumVar(0.0, 1.0, to_string(ctr++));
            action_sum->SetCoefficient(ac, 1);
            policy.insert({*ac_it, ac}); // add to policy to access solution

            auto states_distr = common_data.handler.outcome_probabilities(root->state, *ac_it);

            for (auto it = child_it->children.begin(); it != child_it->children.end(); ++it) {

                MPVariable* st = solver->MakeNumVar(0.0, 1.0, to_string(ctr++));

                // st = ac * delta (3)
                MPConstraint* const ac_st = solver->MakeRowConstraint(0, 0);
                ac_st->SetCoefficient(st, -1);
                double delta = states_distr[it->first];
                ac_st->SetCoefficient(ac, delta);

                LP_policy_rec(it->second.get(), st, ctr, risk_cons, objective, 1, leaves, it->first, payoff);
            }
        }

        objective->SetMaximization();

        return {policy, leaves};
    }

    void LP_policy_rec(state_node_t* node, MPVariable* var, size_t& ctr, MPConstraint* const risk_cons,
                        MPObjective* const objective, size_t node_depth,
                        std::map<S, std::vector<MPVariable*>> leaves, S root_succ, double payoff) {

        payoff += std::pow(common_data.gamma, node_depth - 1) * node->observed_reward;

        if (node->leaf) {

            // objective
            double coef = payoff + std::pow(common_data.gamma, node_depth) * node->rollout_reward;
            objective->SetCoefficient(var, coef);

            // risk
            risk_cons->SetCoefficient(var, node->observed_penalty);

            // alternative risk
            if (node->observed_penalty > 0)
                leaves[root_succ].push_back(var);

            // spdlog::debug("root_succ: {} leaf_payoff: {} p: {}", to_string(root_succ), payoff, node->observed_penalty);
            return;
        }

        MPConstraint* const action_sum = solver->MakeRowConstraint(0, 0); // setting 0 0 for equality could cause rounding problems
        action_sum->SetCoefficient(var, -1); // sum of action prob == prob of parent (2)

        auto& actions = node->actions;
        auto& children = node->children;
        auto child_it = children.begin();
        for (auto ac_it = actions.begin();
             ac_it != actions.end(); ++ac_it, ++child_it) {

            MPVariable* const ac = solver->MakeNumVar(0.0, 1.0, to_string(ctr++)); // x_h,a
            action_sum->SetCoefficient(ac, 1);

            auto states_distr = common_data.handler.outcome_probabilities(node->state, *ac_it);

            for (auto it = child_it->children.begin(); it != child_it->children.end(); ++it) {

                MPVariable* const st = solver->MakeNumVar(0.0, 1.0, to_string(ctr++));

                // st = ac * delta (3)
                MPConstraint* const ac_st = solver->MakeRowConstraint(0, 0);
                ac_st->SetCoefficient(st, -1);
                double delta = states_distr[it->first];
                ac_st->SetCoefficient(ac, delta);

                LP_policy_rec(it->second.get(), st, ctr, risk_cons, objective, node_depth + 1, leaves, root_succ, payoff);
            }
        }
    }

    auto define_LP_risk() {

        //assert(!root.leaf());

        solver->Clear();

        size_t ctr = 0; //counter

        MPObjective* const objective = solver->MutableObjective();

        MPVariable* const r = solver->MakeIntVar(1, 1, "r"); // (1)

        MPConstraint* const action_sum = solver->MakeRowConstraint(0, 0); // setting 0 0 for equality could cause rounding problems
        action_sum->SetCoefficient(r, -1); // sum of action prob == prob of parent (2)

        auto& actions = root->actions;
        auto& children = root->children;
        auto child_it = children.begin();
        for (auto ac_it = actions.begin();
             ac_it != actions.end(); ++ac_it, ++child_it) {

            MPVariable* const ac = solver->MakeNumVar(0.0, 1.0, to_string(ctr++)); // x_h,a
            action_sum->SetCoefficient(ac, 1);

            auto states_distr = common_data.handler.outcome_probabilities(root->state, *ac_it);

            for (auto it = child_it->children.begin(); it != child_it->children.end(); ++it) {

                MPVariable* const st = solver->MakeNumVar(0.0, 1.0, to_string(ctr++));

                // st = ac * delta (3)
                MPConstraint* const ac_st = solver->MakeRowConstraint(0, 0);
                ac_st->SetCoefficient(st, -1);
                double delta = states_distr[it->first];
                ac_st->SetCoefficient(ac, delta);

                LP_risk_rec(it->second.get(), st, ctr, objective);
            }
        }

        objective->SetMinimization();

        return objective;
    }

    void LP_risk_rec(state_node_t* node,
                     MPVariable* const var, size_t& ctr,
                     MPObjective* const objective) {

        if (node->is_leaf()) {

            // objective
            objective->SetCoefficient(var, node->observed_penalty);

            return;
        }                  

        MPConstraint* const action_sum = solver->MakeRowConstraint(0, 0); // setting 0 0 for equality could cause rounding problems
        action_sum->SetCoefficient(var, -1); // sum of action prob == prob of parent (2)

        auto& actions = node->actions;
        auto& children = node->children;
        auto child_it = children.begin();
        for (auto ac_it = actions.begin();
             ac_it != actions.end(); ++ac_it, ++child_it) {

            MPVariable* const ac = solver->MakeNumVar(0.0, 1.0, to_string(ctr++)); // x_h,a
            action_sum->SetCoefficient(ac, 1);

            auto states_distr = common_data.handler.outcome_probabilities(node->state, *ac_it);

            for (auto it = child_it->children.begin(); it != child_it->children.end(); ++it) {

                MPVariable* const st = solver->MakeNumVar(0.0, 1.0, to_string(ctr++));

                // st = ac * delta (3)
                MPConstraint* const ac_st = solver->MakeRowConstraint(0, 0);
                ac_st->SetCoefficient(st, -1);
                double delta = states_distr[it->first];
                ac_st->SetCoefficient(ac, delta);

                LP_risk_rec(it->second.get(), st, ctr, objective);
            }
        } 
    }

    void reset() override {
        spdlog::debug("Reset: {}", name());
        agent<S, A>::reset();
        risk_thd = common_data.risk_thd;
        root = std::make_unique<state_node_t>();
        root->common_data = &common_data;
    }

    std::string name() const override {
        return "ramcp";
    }
};

}
}