#include <gtest/gtest.h>
#include "../../include/containers/Container.h"
#include "../../include/layers/Linear.h"
#include "../../include/layers/RELU.h"

namespace {
    class ContainerTest : public ::testing::Test { };

    TEST(ContainerTest, construction) {
        Linear layer1(1, 2, 0.5);
        RELU layer2;
        Container cont(
                new Linear(2, 5),
                new RELU,
                new Linear(5, 2)
                );
    }
}