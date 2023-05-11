#include <gtest/gtest.h>
#include "../../include/containers/Container.h"

namespace {
    class ContainerTest : public ::testing::Test { };

    TEST(ContainerTest, construction) {
        Container cont(
                BaseModule::make<Linear>(2, 10),
                BaseModule::make<ReLU>(),
                BaseModule::make<Linear>(10, 10),
                BaseModule::make<ReLU>(),
                BaseModule::make<Linear>(10, 2),
                BaseModule::make<Softmax>()
        );
        Eigen::MatrixXf batch1{
                {1, 2, 3, 8, 7, 7, 7},
                {4, 5, 6, 7, 7, 7, 7}
        };
        auto modelOutput1 = cont.forward(batch1);
        Eigen::MatrixXf target1{
                {2, 3, 4, 5, 5, 4, 3},
                {1, 2, 1, 2, 4, 3, 2}
        };
        cont.backward<MSE>(modelOutput1, target1);
    }
}