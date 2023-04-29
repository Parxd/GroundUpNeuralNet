#include <gtest/gtest.h>
#include "../../include/containers/Container.h"
#include "../../include/layers/RELU.h"

namespace {
    class ContainerTest : public ::testing::Test { };

    TEST(ContainerTest, construction) {
        Container cont(
                new Linear(2, 5),
                new RELU(),
                new Linear(5, 6)
        );
        cont.view();

        Container c(
                std::make_unique<Linear>(3, 5).release();
                );
        c.view();
    }
}