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
    using std::to_string;

    template< typename S, typename A >
    class LP_solver {

        environment<S, A>* env;
        std::unique_ptr<MPSolver> solver;
        // S -> occupancy measure
        std::map<S, LinearExpr> occ;
        float risk_thd;
        float gamma = 0.99;
        float gammap = 1;
        LinearExpr total_reward;
        MPObjective* objective;
        size_t ctr = 0;


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

        void set_payoff_objective(S start, LinearExpr& total_penalty) {
            spdlog::debug("Setting LP payoff objective with risk threshold {}", risk_thd);

            // Payoff objective: maximize total reward
            objective = solver->MutableObjective();
            objective->OptimizeLinearExpr(total_reward, true); // true -> maximize;

            // Risk constraint: total penalty <= risk threshold
            spdlog::trace("Total penalty: {}", expr2str(total_penalty));
            spdlog::trace("Total reward: {}", expr2str(total_reward));

            solver->MakeRowConstraint(total_penalty <= risk_thd);
        }

        void construct_LP() {
            solver->Clear();
            occ.clear();

            ctr = 0;
            total_reward = LinearExpr();
            LinearExpr total_penalty = 0;
            S start = env->current_state();
            occ[start] = 1.;

            rec_construct(start, total_penalty);
            set_payoff_objective(start, total_penalty);
        }

        void rec_construct(S parent, LinearExpr& total_penalty, float rew_discount = 1.f, float cost_discount = 1.f) {

            if (env->is_terminal(parent)) { return; }

            spdlog::trace("Setting LP flow for node {}", to_string(parent));
            for ( auto action : env.possible_actions(parent) ) {

                MPVariable* const action_occ = solver->MakeNumVar(0.0, 1.0, "A"+to_string(ctr++));
                occ[parent] -= action_occ;

                auto states_distr = env->outcome_probabilities(parent, action);
                for (auto& [state, prob] : states_distr) {
                    if (!occ.contains(state)) {
                        rec_construct(state, total_penalty, gamma * rew_discount, gammap * cost_discount);
                    }

                    occ[state] += LinearExpr(action_occ) * states_distr[state];
                    auto [rew, cost] = env->get_expected_reward(parent, action, state);
                    total_reward += LinearExpr(action_occ) * states_distr[state] * rew * rew_discount;
                    total_penalty += LinearExpr(action_occ) * states_distr[state] * cost * cost_discount;
                }
            }

            solver->MakeRowConstraint(occ[parent] == 0.f);
        }

        public:

        LP_solver(environment<S, A>& _env, float _risk_thd)
         : env(&_env), solver(std::unique_ptr<MPSolver>(MPSolver::CreateSolver("GLOP"))), risk_thd(_risk_thd) {}

        float solve() {
            construct_LP();
            if (solver->Solve() != MPSolver::OPTIMAL) {
                throw std::runtime_error("Infeasible environment setup");
            }

            return objective->Value();
        }

        void change_thd(float new_thd) {
            risk_thd = new_thd;
        }

        void change_gammas(float _gamma, float _gammap = 1.f) {
            gamma = _gamma;
            gammap = _gammap;
        }

        void change_env(environment<S, A>& _env) {
            env = &_env;
        }
    };
}