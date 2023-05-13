#include <gtest/gtest.h>
#include "../../include/data/Sine.h"

namespace {
    class SineTest : public ::testing::Test {
    };

    TEST(SineTest, construction) {
        auto train = Sine::generate(400000, 0.2, 10, 5, 1.1);
        auto test = Sine::generate(100000,  0.2, 10, 5, 1.1);
    }
}