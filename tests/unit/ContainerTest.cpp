#include <gtest/gtest.h>
#include "../../include/containers/Container.h"
#include "../../include/layers/RELU.h"
#include "../../include/layers/Sigmoid.h"
#include "../../include/layers/Softmax.h"
#include "../../include/losses/MSE.h"

namespace {
    class ContainerTest : public ::testing::Test { };

    TEST(ContainerTest, construction) {
        Container cont(
                BaseModule::make<Linear>(2, 10),
                BaseModule::make<RELU>(),
                BaseModule::make<Linear>(10, 10),
                BaseModule::make<RELU>(),
                BaseModule::make<Linear>(10, 2),
                BaseModule::make<Softmax>()
        );
        Eigen::MatrixXf data{
                {1, 2, 3, 8, 7, 7, 7},
                {4, 5, 6, 7, 7, 7, 7}
        };
        auto modelOutput = cont.forward(data);

        Eigen::MatrixXf target{
                {2, 3, 4, 5, 5, 4, 3},
                {1, 2, 1, 2, 4, 3, 2}
        };
        for (int i = 0; i < 500; ++i) {
            cont.backward(modelOutput, target);
        }
        std::cout << cont.forward(data);
    }
}