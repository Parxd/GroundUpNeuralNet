#include <random>
#include "../../include/data/Sine.h"

Eigen::MatrixXf Sine::generate(int num, float errorSplit = 0.2, float tolerance = 20, int xStretchFactor = 5, float ySquashFactor = 1.1) {
    auto correctCols = int(float(num) * (1 - errorSplit));
    auto incorrectCols = int(float(num) * errorSplit);
    Eigen::MatrixXf data(3, num);

    // Correct data
    data.row(0).head(correctCols) = Eigen::VectorXf::Random(correctCols) * xStretchFactor;
    data.row(1).head(correctCols) = data.row(0).head(correctCols).array().sin() / ySquashFactor +
                                        (Eigen::MatrixXf::Random(1, correctCols) / tolerance).array();
    data.row(2).head(correctCols) = Eigen::VectorXf::Ones(correctCols);

    // Incorrect data
    data.row(0).tail(incorrectCols).setRandom();
    data.row(1).tail(incorrectCols).setRandom();
    data.row(2).tail(incorrectCols) = Eigen::VectorXf::Zero(incorrectCols);

    // Shuffle data
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permX(data.cols());
    permX.setIdentity();
    std::shuffle(permX.indices().data(), permX.indices().data()+permX.indices().size(), eng);
    data = data * permX;

    return data;
}
