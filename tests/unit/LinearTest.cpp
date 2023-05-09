#include <gtest/gtest.h>
#include "../../include/layers/Linear.h"

namespace {
    class LinearTest : public ::testing::Test {
    };

    TEST(LinearTest, LayerConstruction) {
        Eigen::MatrixXf weights{
                {-2, 3,  1,  5, 4},
                {2,  1,  -1, 2, 1},
                {1,  -2, 2,  3, 5}
        };
        Eigen::MatrixXf bias{
                {1},
                {-1},
                {0},
                {0.5},
                {-0.5}
        };
        Linear layer(3, 5);
        layer.setBias(bias);
        layer.setWeight(weights);
        Eigen::MatrixXf w_l{
                {-2, 2,  1},
                {3,  1,  -2},
                {1,  -1, 2},
                {5,  2,  3},
                {4,  1,  5}
        };
        Eigen::MatrixXf b_l{
                {1},
                {-1},
                {0},
                {0.5},
                {-0.5}
        };
        EXPECT_EQ(layer.getWeight(), w_l);
        EXPECT_EQ(layer.getBias(), b_l);
    }

    TEST(LinearTest, forward) {
        Linear layer(3, 5);
        Eigen::MatrixXf w_l{
                {1, 1, 1},
                {2, 2, 2},
                {3, 3, 3},
                {4, 5, 5},
                {5, 5, 5}
        };
        Eigen::MatrixXf b_l{
                {1},
                {1},
                {1},
                {1},
                {1}
        };
        layer.setWeight(w_l);
        layer.setBias(b_l);
        Eigen::MatrixXf inputs{
                {1, 2, 3, 4, 5, 6, 7, 8, 9},
                {1, 2, 3, 4, 5, 6, 7, 8, 9},
                {1, 2, 3, 4, 5, 6, 7, 8, 9}
        };
        Eigen::MatrixXf newInputs{
                {4,  7,  10, 13, 16, 19, 22,  25,  28},
                {7,  13, 19, 25, 31, 37, 43,  49,  55},
                {10, 19, 28, 37, 46, 55, 64,  73,  82},
                {15, 29, 43, 57, 71, 85, 99,  113, 127},
                {16, 31, 46, 61, 76, 91, 106, 121, 136}
        };
        // Check for correct operations on input
        EXPECT_EQ(layer.forward(inputs), newInputs);
    }

    TEST(LinearTest, backward1) {
        /**
         * Simulating backpropagation
         * We need to manually create the matrix containing the derivative of our loss function
         * with respect to layer sub (l + 1)'s (a.k.a next layer's) activation, which in our
         * container module, would be some kind of activation function.
         *
         * Thus, we need a matrix where each value is the cost-to-activation gradient (a.k.a
         * the derivative of our loss function) multiplied by the derivative of the activation function of Z^L.
         * Z^L is the raw output from this linear layer & multiplied here refers to Hadamard product operation.
         */
        Linear layer(2, 3);
        Eigen::MatrixXf input{
                {2, 3, 4, 5},
                {5, 6, 7, 8},
        };
        Eigen::MatrixXf lossToNextActivationDerivative{
                {2, 3, 4, 5},
                {5, 6, 7, 8},
                {8, 9, 1, 2}
        };
        layer.forward(input);
        auto result = layer.backward(lossToNextActivationDerivative);

    }

    TEST(LinearTest, backward2)
    {
        Linear layer(2, 4);
        Eigen::MatrixXf weights{
                {0.5f, 0.1f, -0.5f, 0.1f},
                {0.09f, -0.5f, 0.1f, 0.09f},
        };
        weights.transposeInPlace();
        Eigen::MatrixXf bias{
            {-0.2f},
            {1.f},
            {0.f},
            {-0.5f}
        };
        layer.setWeight(weights);
        layer.setBias(bias);
        Eigen::MatrixXf forwardInput{
                {-9.f, -5.f},
                {1.f, -3.f},
                {-2.f, 7.f},
        };
        forwardInput.transposeInPlace();
        layer.forward(forwardInput);

        Eigen::MatrixXf lossToNextActivationDerivative{
                {2, 3, 4, 5},
                {5, 6, 7, 8},
                {8, 9, 1, 2}
        };
        lossToNextActivationDerivative.transposeInPlace();
        // Backpropagate the simulated loss derivative calculated
        auto result = layer.backward(lossToNextActivationDerivative);

    }
}
