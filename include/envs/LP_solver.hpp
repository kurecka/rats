#pragma once

#include "env.hpp"
#include "rand.hpp"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_expr.h"
#include <string>
#include <vector>
#include <map>
#include <set>

namespace rats {

    using namespace operations_research;

    template< typename S, typename A >
    class LP_solver {

        environment<S, A>* env;
        std::unique_ptr<MPSolver> solver;
        std::map<S, std::pair<LinearExpr, LinearExpr>> occs;
        float risk_thd;
        float gamma = 0.99;
        float gammap = 1;
        LinearExpr total_reward;
        MPObjective* objective;
        size_t ctr = 0;

        public:

        LP_solver(environment<S, A>& _env, float _risk_thd)
         : env(&_env), solver(std::unique_ptr<MPSolver>(MPSolver::CreateSolver("GLOP"))), risk_thd(_risk_thd) {}

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

        void construct_LP() {
            solver.clear();
            occs.clear();

            ctr = 0;
            total_reward = LinearExpr();
            LinearExpr total_penalty = 0;
            S begin = env->current_state();
            occs[begin] = {1., 1.};

            rec_construct(begin, total_penalty)
        }

        void rec_construct(S parent, LinearExpr total_penalty) {

            if (env->is_terminal(parent)) { return; }

            spdlog::trace("Setting LP flow for node {}", to_string(parent));
            for ( auto action : end.possible_actions(parent) ) {

                MPVariable* const action_occ = solver->MakeNumVar(0.0, 1.0, "A"+to_string(ctr++));
                
            }
        }

        void solve();

        // change_thd

        // change_gammas

        // change_env
    };
}