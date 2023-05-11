#include <iostream>
#include "../../include/losses/CE.h"

float CE::forward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target) {
    float avgError = 0;
    auto numClasses = float(target.rows());
    for (int i = 0; i < target.cols(); ++i)
    {
        float sigma = 0;
        sigma += (target.col(i).array() * pred.col(i).array().log()).sum();
        sigma *= (-1 / numClasses);
        avgError += sigma;
    }
    avgError /= float(target.cols());
    return avgError;
}

Eigen::MatrixXf CE::backward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target) {
    Eigen::MatrixXf output = -target.array() / (pred.array() + 0.000001);
    return output;
}