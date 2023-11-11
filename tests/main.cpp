#include "unittest.hpp"
#include "test_example.hpp"
#include "test_agents.hpp"

std::list<UnitTest*> UnitTest::testList;

int main(){
    register_example_tests();
    register_agents_tests();
    
    RunTests()
	
    return 0;
}