#include <gtest/gtest.h>
#include "../../include/data/Sine.h"

namespace {
    class SineTest : public ::testing::Test {
    };

    TEST(SineTest, construction) {
        auto data = Sine::generate(20, 0.2, 30, 5, 1.1);
        for (int i = 0; i < data.cols(); ++i)
        {
            if (data.row(2).col(i).data()[0] == 1.f)
            {
                EXPECT_NEAR(std::sin(data.row(0).col(i).data()[0]), data.row(1).col(i).data()[0], 0.1);
            }
        }
    }
}