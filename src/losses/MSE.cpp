#include "../../include/losses/MSE.h"

float MSE::forward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target)
{
    Eigen::MatrixXf squaredErrorMatrix = (pred - target).array().square();
    auto error = squaredErrorMatrix.sum() / float(target.cols());
    return error;
}

Eigen::MatrixXf MSE::backward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target)
{
    Eigen::MatrixXf dLA = 2 * (pred - target) / target.cols();
    return std::move(dLA);
}