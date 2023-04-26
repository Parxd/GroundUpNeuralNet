#include <gtest/gtest.h>
#include "../../include/layers/Linear.h"

namespace {
    class LinearTest : public ::testing::Test {
    };

    TEST(LinearTest, forward1) {
        // Linear setup
        int inputs = 3;
        int outputs = 5;
        float LR = 0.01;
        Eigen::MatrixXf weights{
                {-2, 3,  1},
                {2,  1,  -1},
                {1,  -2, 2},
                {-1, 0,  -3},
                {1,  2,  1}
        };
        Eigen::MatrixXf bias{
                {1},
                {-1},
                {0},
                {1},
                {-0.5}
        };
        Linear layer(inputs, outputs, LR);

        // Set inputs
        // NOTE: This assumes the first layer of the network will take in a transposed vector (.transposeInPlace() on source data)
        Eigen::MatrixXf input{
                {0.5,  0.6},
                {-0.5, 0.8},
                {1,    0.7}
        };
        Eigen::MatrixXf answer{
                {-0.5},
                {-1.5},
                {3.5},
                {-2.5},
                {0}
        };
        Eigen::MatrixXf result;
        layer.setWeight(weights);
        layer.setBias(bias);
    }
};