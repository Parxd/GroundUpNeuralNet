#include <gtest/gtest.h>
#include "../../include/layers/Linear.h"

namespace {
    class LinearTest : public ::testing::Test { };

    TEST(LinearTest, forward1) {
        // Linear setup
        int inputs = 3;
        int outputs = 5;
        float LR = 0.01;
        Eigen::MatrixXf weights{
                {-2, 3,  1, 5, 4},
                {2,  1,  -1, 2, 1},
                {1,  -2, 2, 3, 5}
        };
        Eigen::MatrixXf bias{
                {1}, {-1}, {0}, {0.5}, {-0.5}
        };
        Linear layer(inputs, outputs, LR);

        layer.setBias(bias);
        layer.setWeight(weights);
    }
}