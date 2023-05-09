#include <gtest/gtest.h>
#include "../../include/layers/Softmax.h"

namespace {
    class SoftmaxTest : public ::testing::Test {
    };

    TEST(SoftmaxTest, forward) {
        Eigen::MatrixXf input{
                {8, 8},
                {5, 5},
                {0, 0}
        };
        Softmax lol;
        lol.forward(input);
    }
}