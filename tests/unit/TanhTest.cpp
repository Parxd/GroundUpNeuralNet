#include <gtest/gtest.h>
#include "../../include/layers/Tanh.h"

namespace {
    class TanhTest : public ::testing::Test {
    };

    TEST(TanhTest, forward1) {
        Eigen::MatrixXf input(2, 2);
        input << 1, 2, 3, 4;

        Eigen::MatrixXf expected_output(2, 2);
        expected_output << std::tanh(1), std::tanh(2), std::tanh(3), std::tanh(4);

        Tanh tanh;
        Eigen::MatrixXf actual_output = tanh.forward(input);

        // Check that actual output matches expected output
        ASSERT_TRUE(actual_output.isApprox(expected_output));
    }

    TEST(TanhTest, forward2) {
        Tanh t;
        Eigen::MatrixXf input(2, 3);
        input << 0.5f, -1.0f, 2.0f,
                -0.5f, 1.0f, -2.0f;
        Eigen::MatrixXf expected_output(2, 3);
        expected_output << 0.462117f, -0.761594f, 0.964028f,
                        -0.462117f, 0.761594f, -0.964028f;
        Eigen::MatrixXf output = t.forward(input);
        ASSERT_TRUE(output.isApprox(expected_output, 1e-5f));
    }
}