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

    }
}