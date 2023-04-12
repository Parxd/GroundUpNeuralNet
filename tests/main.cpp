#include <iostream>
#include <gtest/gtest.h>
#include "../include/layers/Linear.h"
#include "../include/layers/activations/RELU.h"
#include "../include/layers/activations/Sigmoid.h"
#include "../include/layers/activations/Softmax.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}
