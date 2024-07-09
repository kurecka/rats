#include "unittest.hpp"
#include "test_example.hpp"
#include "test_pareto_curves.hpp"

std::list<UnitTest*> UnitTest::testList;

int main(){
    register_example_tests();
    register_pareto_curves_tests();
    
    RunTests()
	
    return 0;
}