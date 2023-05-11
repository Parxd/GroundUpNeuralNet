#include <gtest/gtest.h>
#include "../../include/losses/CE.h"

namespace {
    class ContainerTest : public ::testing::Test { };

    TEST(CETest, forward) {
        Eigen::MatrixXf pred{
                {0.9,  0.01},
                {0.05, 0.98},
                {0.05, 0.01}
        };
        Eigen::MatrixXf target{
                {1, 0},
                {0, 1},
                {0, 0}
        };
        auto result = CE::forward(pred, target);
        float answer = 0.020927;
        EXPECT_NEAR(result, answer, 0.00001);
    }

    TEST(CETest, backward) {
        Eigen::MatrixXf pred{
                {0.09},
                {0.50},
                {0.40},
                {0.01}
        };
        Eigen::MatrixXf target{
                {0},
                {0},
                {0},
                {1}
        };
        std::cout << CE::forward(pred, target);
        auto errorDerivative = CE::backward(pred, target);
    }
}
