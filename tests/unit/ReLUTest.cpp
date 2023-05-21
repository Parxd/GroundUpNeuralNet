#include <gtest/gtest.h>
#include "../../include/layers/ReLU.h"

namespace {
    class ReLUTest : public ::testing::Test {
    };

    TEST(ReLUTest, forward1) {
        Eigen::MatrixXf input(2, 2);
        input << -1, 2, -3, 4;

        Eigen::MatrixXf expected_output(2, 2);
        expected_output << 0, 2, 0, 4;

        ReLU relu;
        Eigen::MatrixXf actual_output = relu.forward(input);

        // Check that actual output matches expected output
        ASSERT_TRUE(actual_output.isApprox(expected_output));
    }

    TEST(ReLUTest, forward2) {
        Eigen::MatrixXf input(3, 3);
        input << -2, 0, 3, 1, -4, 2, 5, -1, -6;

        Eigen::MatrixXf expected_output(3, 3);
        expected_output << 0, 0, 3, 1, 0, 2, 5, 0, 0;

        ReLU relu;
        Eigen::MatrixXf actual_output = relu.forward(input);

        // Check that actual output matches expected output
        ASSERT_TRUE(actual_output.isApprox(expected_output));
    }
}