#include <gtest/gtest.h>
#include "../../include/layers/Linear.h"

class LinearTest : public ::testing::Test { };

TEST(LinearTest, test0)
{
    // Linear setup
    int inputs = 3;
    int outputs = 5;
    float LR = 0.01;
    Eigen::MatrixXf weights{
        {-2, 3, 1},
        {2, 1, -1},
        {1, -2, 2},
        {-1, 0, -3},
        {1, 2, 1}
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
    // NOTE: This assumes the first layer of the network will take in a tranposed vector (.transposeInPlace() on source data)
    Eigen::MatrixXf input{
        {0.5}, 
        {-0.5}, 
        {1}
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

    // Weight structure test
    EXPECT_EQ(layer.getWeight().rows(), outputs);
    EXPECT_EQ(layer.getWeight().cols(), inputs);
    // Bias structure test
    EXPECT_EQ(layer.getBias().rows(), outputs);
    EXPECT_EQ(layer.getBias().cols(), 1);
    // Forward test
    layer.forward(input, result);
    EXPECT_EQ(answer, result);
}