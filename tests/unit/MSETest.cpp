#include <gtest/gtest.h>
#include "../../include/losses/MSE.h"

namespace {
    class MSETest : public ::testing::Test {
    };

    TEST(MSETest, basics) {
        Eigen::MatrixXf modelOutput{
            {2, 4, 1},
            {5, 3, 6},
            {8, 2, 7}
        };
        Eigen::MatrixXf target{
            {3, 5, 2},
            {6, 6, 6},
            {8, 2, 1}
        };
        float error = MSE::forward(modelOutput, target);
        auto errorDerivative = MSE::backward(modelOutput, target);
    }
}