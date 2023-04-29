#include <gtest/gtest.h>
#include "../../include/containers/Container.h"
#include "../../include/layers/RELU.h"

namespace {
    class ContainerTest : public ::testing::Test { };

    TEST(ContainerTest, construction) {
        Container cont(
                BaseModule::make<Linear>(3, 5),
                BaseModule::make<RELU>()
                );
        cont.view();

        std::cout << std::endl;

        Container cont2(
                new Linear(3, 5),
                new RELU,
                new Linear(5, 2)
                );
        cont2.view();
    }
}