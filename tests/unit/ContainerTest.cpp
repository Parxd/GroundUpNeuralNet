#include <gtest/gtest.h>
#include "../../include/containers/Container.h"
#include "../../include/layers/RELU.h"

namespace {
    class ContainerTest : public ::testing::Test { };

    TEST(ContainerTest, construction) {
        Container cont(
                new Linear(3, 5),
                new RELU,
                new Linear(5, 6)
                );
        cont.view();

        std::cout << "\n";
        std::vector<std::unique_ptr<BaseModule>> vector;
        vector.push_back(std::make_unique<Linear>(784, 23));
        Container cont2(vector);
        cont2.view();
    }
}