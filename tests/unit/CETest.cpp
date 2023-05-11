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
        float answer = 0.06;
        EXPECT_NEAR(result, answer, 0.01);
    }

    TEST(CETest, backward) {
        Eigen::MatrixXf pred{
                {0.98, 0.01, 0.01},
                {0.01, 0.01, 0.01},
                {0.01, 0.98, 0.98},
        };
        Eigen::MatrixXf target{
                {1, 0, 0},
                {0, 0, 0},
                {0, 1, 1}
        };
        CE::forward(pred, target);
        auto errorDerivative = CE::backward(pred, target);
    }
}
