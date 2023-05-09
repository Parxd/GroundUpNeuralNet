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
                BaseModule::make<Linear>(2, 5),
                BaseModule::make<Sigmoid>(),
                BaseModule::make<Linear>(5, 3),
                BaseModule::make<RELU>()
                );
        Eigen::MatrixXf data{
                {1, 2, 3, 8},
                {4, 5, 6, 7}
        };
        auto modelOutput = cont.forward(data);

        Eigen::MatrixXf target{
                {2, 3, 4, 5},
                {1, 1, 1, 2},
                {1, 2, 2, 3}
        };
        cont.backward(modelOutput, target);
    }
}