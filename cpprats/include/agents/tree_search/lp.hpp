#pragma once

#include "tree_search.hpp"
#include "string_utils.hpp"
#include "rand.hpp"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_expr.h"
#include <algorithm>
#include <string>
#include <vector>
#include <map>


namespace rats::ts::lp {

using namespace operations_research;

template <typename SN>
class tree_lp_solver {
    using S = typename SN::S;
    using A = typename SN::A;
    using AN = typename SN::action_node_t;
    using DATA = typename SN::DATA;

private:
    std::unique_ptr<MPSolver> solver;

    /*
    The following variables are defined after set_flow(root) is called:
    policy: probability of each action in the root node
    subtree_penalties: expected penalty of each subtree rotted at each action-state pair (i.e. they sum up to the total penalty)
    ctr: counter for variable names
    total_reward: each probability flow variable is multiplied by the coreesponding reward
    objective: objective function
        - if set_payoff_objective is called, it is the total reward and maximized
        - if set_penalty_objective is called, it is the total penalty and minimized
    */
    std::map<A, MPVariable*> policy;
    std::map<std::pair<A, S>, LinearExpr> subtree_penalties;
    size_t ctr = 0;
    LinearExpr total_reward;
    MPObjective* objective;
    DATA* common_data;
    S root_state;

private:
    std::string expr2str(const LinearExpr& expr) {
        std::string str;
        for (auto& [var, coef] : expr.terms()) {
            str += fmt::format("{} * {} + ", coef, var->name());
        }
        return str + to_string(expr.offset());
    }

    std::string const2str(const LinearRange& expr) {
        std::string str = to_string(expr.lower_bound()) + " <= ";
        str += expr2str(expr.linear_expr());
        str += " <= " + to_string(expr.upper_bound());
        return str;
    }

    void set_payoff_objective(const SN* root, double risk_thd) {
        spdlog::debug("Setting LP payoff objective with risk threshold {}", risk_thd);
        set_flow(root);

        // Payoff objective: maximize total reward
        objective = solver->MutableObjective();
        objective->OptimizeLinearExpr(total_reward, true); // true -> maximize;

        // Risk constraint: total penalty <= risk threshold
        LinearExpr total_penalty = LinearExpr();
        for (auto& [as, expr] : subtree_penalties) {
            total_penalty += expr;
        }

        spdlog::trace("Total penalty: {}", expr2str(total_penalty));
        spdlog::trace("Total reward: {}", expr2str(total_reward));

        solver->MakeRowConstraint(total_penalty <= risk_thd);
    }

    void set_penalty_objective(const SN* root) {
        spdlog::debug("Setting LP penalty objective");
        set_flow(root);

        LinearExpr total_penalty = LinearExpr();
        for (auto& [as, expr] : subtree_penalties) {
            total_penalty += expr;
        }

        objective = solver->MutableObjective();
        objective->OptimizeLinearExpr(total_penalty, false); // false -> minimize;
    }

    /**
     * Set probability flow LP and prepare necessary sums.
    */
    void set_flow(const SN* root) {
        solver->Clear();
        policy.clear();
        subtree_penalties.clear();
        ctr = 0;
        total_reward = LinearExpr();

        set_flow_rec(root);
    }

    void set_flow_rec(
        const SN* parent,  // Parent node
        LinearExpr parent_flow = 1.,  // Flow from the parent node
        float value_discount = 1.,  // Discount factor for the value
        float penalty_discount = 1.,  // Discount factor for the penalty
        LinearExpr* subtree_penalty = nullptr  // Pointer to the subtree penalty variable of the current node
    ) {
        // If subtree_penalty does not exist, this is the root node.
        bool is_LP_root = !subtree_penalty;
        for (auto& action_node : parent->children) {
            // Create action flow variable: non-negative value
            MPVariable* const action_flow = solver->MakeNumVar(0.0, 1.0, "A"+to_string(ctr++));
            parent_flow -= action_flow;
            A action = action_node.action;

            // In the root node, set the action flow as the policy variable
            if (is_LP_root) {
                policy.insert({action, action_flow});
            }

            auto states_distr = common_data->predictor.predict_probs(parent->state, action);
            for (auto& [state, state_node] : action_node.children) {
                // Create state flow variable: non-negative value
                MPVariable* state_flow = solver->MakeNumVar(0.0, 1.0, "S"+to_string(ctr++));

                // state_flow = action_flow * p(s | a)
                solver->MakeRowConstraint(state_flow == LinearExpr(action_flow) * states_distr[state]);

                // Add discounted incoming reward to total reward
                total_reward += LinearExpr(state_flow) * state_node->observed_reward * value_discount;

                // In the root node, create the subtree penalty variable
                if (is_LP_root) {
                    subtree_penalties[{action, state}] = LinearExpr();
                    subtree_penalty = &subtree_penalties[{action, state}];
                }
                // Add discounted penalty to the subtree penalty
                (*subtree_penalty) += LinearExpr(state_flow) * state_node->observed_penalty * penalty_discount;

                if (!state_node->is_leaf_state()) {
                    set_flow_rec(
                        state_node.get(),
                        LinearExpr(state_flow),
                        value_discount * common_data->gamma,
                        penalty_discount * common_data->gammap,
                        subtree_penalty
                    );
                } else {
                    // Add discounted rollout estimate to total reward
                    total_reward += LinearExpr(state_flow) * value_discount * common_data->gamma * state_node->rollout_reward;
                    // Add non-discounted rollout penalty to the subtree penalty
                    (*subtree_penalty) += LinearExpr(state_flow) * penalty_discount * common_data->gammap * state_node->rollout_penalty;
                }
            }
        }

        // Parent state flow minus the sum of child action flows (already included in `parent_flow`) should be zero
        solver->MakeRowConstraint(parent_flow == 0.f);
    }

public:
    tree_lp_solver() : solver(std::unique_ptr<MPSolver>(MPSolver::CreateSolver("GLOP"))) {}

    /**
     * Compute the optimal policy for the given tree based on the given risk threshold, and estimated transition probabilities.
     * Keep the byproduct of the computation for future use.
    */
    size_t get_action(const SN* root, double risk_thd) {
        common_data = root->common_data;
        root_state = root->state;
        set_payoff_objective(root, risk_thd);

        if (solver->Solve() != MPSolver::OPTIMAL) {
            spdlog::debug("LP solver failed to find a feasible solution. Relaxing the risk threshold.");
            // If unfiesable, relax the risk threshold to the least feasible value
            set_penalty_objective(root);
            solver->Solve();
            double relaxed_thd = objective->Value();
            spdlog::debug("Relaxed risk threshold: {}", relaxed_thd);

            // Solve again with the relaxed threshold
            set_payoff_objective(root, relaxed_thd + 1e-6);
            solver->Solve();
        }
        spdlog::debug("Objective value: {}", objective->Value());

        // Collect the root node's policy probabilities
        std::vector<double> policy_distr;
        std::string policy_str = "";
        for (auto& [a, var] : policy) {
            policy_distr.emplace_back(var->solution_value());
            policy_str += fmt::format("{}: {:.2f}, ", to_string(a), var->solution_value());
        }
        spdlog::debug("LP policy: {}", policy_str);

        // Sample an action index from the policy
        return rng::custom_discrete(policy_distr);
    }

    /**
     * Compute the maximum penalty threshold after descent under (a, s) so that the constraint is feasible on expectation.
     * 
     * It should hold that p(a,s) * (received_penalty + gamma_penalty * new_thd) + alt_penalty = old_thd
    */
    float update_threshold(float thd, A a, S s, float received_penalty) {
        double branch_budget = thd;
        for (auto& [as, expr] : subtree_penalties) {
            auto [subtree_action, subtree_state] = as;
            if (subtree_action != a || subtree_state != s) {
                branch_budget -= expr.SolutionValue();
            }
        }
        auto states_distr = common_data->predictor.predict_probs(root_state, a);
        float transition_prob = policy[a]->solution_value() * states_distr[s];

        float descendent_budget = (branch_budget / transition_prob - received_penalty) / common_data->gammap;
        return std::clamp(descendent_budget, 0.0f, common_data->max_disc_penalty);
    }
};

} // namespace rats::ts
