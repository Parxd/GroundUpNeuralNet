#include <iostream>
#include <gtest/gtest.h>
#include "../include/layers/Linear.h"
#include "../include/layers/activations/RELU.h"

int main(int argc, char** argv) {
    // ::testing::InitGoogleTest(&argc, argv);
    // return RUN_ALL_TESTS();
    Linear layer(3, 5, 0.001);
    std::cout << layer.description() + " <--- 0\n";

    return 0;
}
