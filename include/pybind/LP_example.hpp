#pragma once

#include <ortools/linear_solver/linear_solver.h>
#include <spdlog/spdlog.h>
#include <memory>
#include "pybind/pybind.hpp"


namespace rats::py::example {
    void solve_lp() {
        using namespace operations_research;

        // OR-Tools LP solver
        std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
        if (!solver) {
            spdlog::warn("Could not create solver SCIP");
            return;
        }

        const double infinity = solver->infinity();
        // x and y are non-negative variables.
        MPVariable* const x = solver->MakeNumVar(0.0, infinity, "x");
        MPVariable* const y = solver->MakeNumVar(0.0, infinity, "y");
        spdlog::info("Number of variables = {}",  solver->NumVariables());

        // x + 2*y <= 14.
        MPConstraint* const c0 = solver->MakeRowConstraint(-infinity, 14.0);
        c0->SetCoefficient(x, 1);
        c0->SetCoefficient(y, 2);

        // 3*x - y >= 0.
        MPConstraint* const c1 = solver->MakeRowConstraint(0.0, infinity);
        c1->SetCoefficient(x, 3);
        c1->SetCoefficient(y, -1);

        // x - y <= 2.
        MPConstraint* const c2 = solver->MakeRowConstraint(-infinity, 2.0);
        c2->SetCoefficient(x, 1);
        c2->SetCoefficient(y, -1);
        spdlog::info("Number of constraints = {}",  solver->NumConstraints());

        // Objective function: 3*x + 4*y.
        MPObjective* const objective = solver->MutableObjective();
        objective->SetCoefficient(x, 3);
        objective->SetCoefficient(y, 4);
        objective->SetMaximization();

        const MPSolver::ResultStatus result_status = solver->Solve();
        // Check that the problem has an optimal solution.
        if (result_status != MPSolver::OPTIMAL) {
            spdlog::error("The problem does not have an optimal solution!");
        }

        // Print the solution.
        spdlog::info("Solution:");
        spdlog::info("Optimal objective value = {}", objective->Value());
        spdlog::info("{} = {}", x->name(), x->solution_value());
        spdlog::info("{} = {}", y->name(), y->solution_value());
    }

    void register_lp_example(py::module& m) {
        m.def("solve_lp", &solve_lp, "Solve a linear program");
    }
} // namespace rats::example
