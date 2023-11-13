#pragma once

#include <list>
#include <iostream>
#include <spdlog/spdlog.h>

class UnitTest {
    protected:
        UnitTest() {}
        UnitTest(std::string name): name{name} {}

    public:
        bool isTrue{true};
        static std::list<UnitTest*> testList;
        std::string name;
        std::string msg;
        
        virtual ~UnitTest() {}
        virtual void runFunc() {}

        static UnitTest& getInstance() {
            static UnitTest uTest;
            return uTest;
        }
        
        void runTests();

        template<typename T>
        inline bool expectEQ(const T& arg1, const T& arg2, std::string file, size_t line) {
            bool isTrue{arg1 == arg2};
            if(!isTrue) {
                msg = file + ":" + std::to_string(line) + " ";
                msg += "ExpectEQ: " + std::to_string(arg1) 
                    + " != " + std::to_string(arg2);
            }
            return isTrue;
        }

        template<typename T1, typename T2, typename T3>
        inline bool areClose(const T1& arg1, const T2& arg2, const T3& tol, std::string file, size_t line) {
            bool isTrue{std::abs(arg1 - arg2) < tol};
            if(!isTrue) {
                msg = file + ":" + std::to_string(line) + " ";
                msg += "AreClose: " + std::to_string(arg1) 
                    + " !~ " + std::to_string(arg2);
                msg += " +- " + std::to_string(tol) + ")";
            }
            return isTrue;
        }

};

void UnitTest::runTests()
{
    // Variables to track passed and failed tests
    int passed{0};
    int failed{0};

    if(testList.empty()) {
        std::cout << "No tests registered!!" << std::endl;
        return;
    }

    for(UnitTest* x: testList) {
        x->runFunc();
        x->isTrue ? ++passed : ++failed;
        if (x->isTrue) {
            std::cout << x->name << " PASS" << std::endl;
        } else {
            std::cout << x->name << " FAIL" << std::endl;
            std::cout << x->msg << std::endl;
        }
        spdlog::set_level(spdlog::level::info);
    }
    
    // Print the number of passed and failed tests
    std::cout << "Total tests:\t" << passed + failed << std::endl;
    std::cout << "Passed tests:\t" << passed << std::endl;
    std::cout << "Failed tests:\t" << failed << std::endl;
}

#define DeclareTest(Module, TestName)                             \
    class Test_##Module##_##TestName: public UnitTest             \
{                                                                 \
    Test_##Module##_##TestName(): UnitTest(#Module "_" #TestName){}                    \
    public:                                                       \
        static Test_##Module##_##TestName* getInstance()          \
    {                                                             \
        static Test_##Module##_##TestName testClass;              \
        return &testClass;                                        \
    }                                                             \
        void runFunc() override;                                  \
};

#define DefineTest(Module, TestName)                              \
void Test_##Module##_##TestName::runFunc()

#define UTest(Module, TestName)                                \
DeclareTest(Module, TestName)                                 \
DefineTest(Module, TestName)


#define RegisterTest(Module, TestName)                    \
    UnitTest::getInstance().testList.push_back(           \
            Test_##Module##_##TestName::getInstance());

#define RunTests()                                        \
    UnitTest::getInstance().runTests();
    
#define ExpectEQ(arg1, arg2)                              \
    isTrue &= expectEQ(arg1, arg2, __FILE__, __LINE__); if (!isTrue) return;


#define AreClose(arg1, arg2)                              \
    isTrue &= areClose(arg1, arg2, 1e-6f, __FILE__, __LINE__); if (!isTrue) return;

#define AreCloseEps(arg1, arg2, eps)                      \
    isTrue &= areClose(arg1, arg2, eps, __FILE__, __LINE__); if (!isTrue) return;