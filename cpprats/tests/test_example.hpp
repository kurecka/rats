#pragma once
#include "unittest.hpp"

UTest(example, test1) {
    ExpectEQ(1, 1);
}

void register_example_tests() {
    RegisterTest(example, test1)
}
