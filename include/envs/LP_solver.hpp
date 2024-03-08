#pragma once

#include "rand.hpp"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_expr.h"
#include <string>
#include <vector>
#include <map>

namespace rats {

    using namespace operations_research;

    class LP_solver {
        public:

        std::unique_ptr<MPSolver> solver;

        LP_solver() : solver(std::unique_ptr<MPSolver>(MPSolver::CreateSolver("GLOP"))) {}
        
    };
}