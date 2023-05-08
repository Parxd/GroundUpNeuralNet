#include <gtest/gtest.h>
#include "../../include/layers/Sigmoid.h"

namespace {
    class SigmoidTest : public ::testing::Test {
    };

    TEST(SigmoidTest, forward) {
        Sigmoid sigmoidLayer;
        Eigen::MatrixXf input{
                {1,   2,   3},
                {4,   5,   6},
                {0.5, 1.2, 6.7}
        };
        auto forwardResult = sigmoidLayer.forward(input);
        Eigen::MatrixXf target{
                {0.7311, 0.8808, 0.9526},
                {0.9820, 0.9933, 0.997},
                {0.6225, 0.7685, 0.9987}
        };
        EXPECT_TRUE(forwardResult.isApprox(target, 0.001));
    }

    TEST(SigmoidTest, backward) {
        Sigmoid sigmoidLayer;
        Eigen::MatrixXf input{
                {1,   2,   3},
                {4,   5,   6},
                {0.5, 1.2, 6.7}
        };
        sigmoidLayer.forward(input);

        // Simulated MSE derivative matrix
        Eigen::MatrixXf errorDerivative{
                {0.5, 0.8, 0.9},
                {0.8, 0.7, 0.7},
                {0.5, 0.4, 0.6}
        };
        auto backwardResult = sigmoidLayer.backward(errorDerivative);
        Eigen::MatrixXf backwardTarget{
                {0.1966, 0.1049, 0.0451},
                {0.0176, 0.0066, 0.0024},
                {0.2350, 0.1778, 0.0012}
        };
        backwardTarget.array() *= errorDerivative.array();
        EXPECT_TRUE(backwardResult.isApprox(backwardTarget, 0.001));
    }
}