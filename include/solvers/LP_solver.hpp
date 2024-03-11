#pragma once

#include "envs/env.hpp"
#include "rand.hpp"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_expr.h"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <limits>

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
        float gamma = 0.99; // has to be < 1
        float gammap =  1;
        LinearExpr total_reward;
        MPObjective* objective;
        size_t ctr = 0;
        const double infinity = solver->infinity();

        // for debug purposes only (policy)
        std::vector<MPVariable*> vars;


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
            // spdlog::trace("Total penalty: {}", expr2str(total_penalty));
            // spdlog::trace("Total reward: {}", expr2str(total_reward));

            solver->MakeRowConstraint(total_penalty <= risk_thd);
        }

        void construct_LP() {
            solver->Clear();
            occ.clear();

            ctr = 0;
            total_reward = LinearExpr();
            LinearExpr total_penalty = 0.;
            S start = env->current_state();
            occ[start] = 1.;

            rec_construct(start, start, total_penalty);

            // set state conversation constraints
            for (auto& [s, lexpr] : occ) {
                if (!env->is_terminal(s)) {
                    // spdlog::trace("occ cons {}", expr2str(lexpr));
                    solver->MakeRowConstraint(lexpr == 0.f);
                }
            }

            set_payoff_objective(start, total_penalty);
        }

        void rec_construct(S start, S parent, LinearExpr& total_penalty, float rew_discount = 1.f, float cost_discount = 1.f) {

            if (env->is_terminal(parent)) { return; }

            // spdlog::trace("Setting LP flow for node {}", to_string(parent));
            for ( auto action : env->possible_actions(parent) ) {

                MPVariable* const action_occ = solver->MakeNumVar(0.0, infinity, "A"+to_string(ctr++));
                occ[parent] -= action_occ;

                vars.push_back(action_occ);

                auto states_distr = env->outcome_probabilities(parent, action);
                for (auto& [state, prob] : states_distr) {
                    if (occ.find(state) == occ.end()) {
                        rec_construct(start, state, total_penalty, gamma * rew_discount, gammap * cost_discount);
                    }

                    occ[state] += LinearExpr(action_occ) * states_distr[state] * gamma;
                    auto [rew, cost] = env->get_expected_reward(parent, action, state);
                    total_reward += LinearExpr(action_occ) * states_distr[state] * rew;
                    total_penalty += LinearExpr(action_occ) * states_distr[state] * cost * cost_discount / rew_discount;

                    // spdlog::trace("Add at transition {}, {}, {}: prob: {}, rew {}, cost {}, action_var: {}",
                    //  to_string(parent), to_string(action), to_string(state), to_string(states_distr[state]), to_string(rew), to_string(cost), expr2str(LinearExpr(action_occ)));
                }
            }
        }

        public:

        LP_solver(environment<S, A>& _env, float _risk_thd)
         : env(&_env), solver(std::unique_ptr<MPSolver>(MPSolver::CreateSolver("GLOP"))), risk_thd(_risk_thd) {}

        float solve() {
            construct_LP();

            // for (auto& [s, e] : occ) {
            //     spdlog::debug("state: {}, occ: {}", to_string(s), expr2str(e));
            // }

            if (solver->Solve() != MPSolver::OPTIMAL) {
                throw std::runtime_error("Infeasible environment setup");
            }

            // for (auto& var : vars) {
            //     spdlog::debug("var: {}, sol value: {}", expr2str(LinearExpr(var)), to_string(var->solution_value()));
            // }

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
