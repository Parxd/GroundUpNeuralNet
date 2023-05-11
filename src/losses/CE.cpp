#include "../../include/losses/CE.h"

float CE::forward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target) {
    double sum = 0;
    for (int i = 0; i < target.cols(); ++i)
    {
        sum += log((target.col(i).array() * pred.col(i).array()).sum());
    }
    return float(-sum / float(target.cols()));
}

Eigen::MatrixXf CE::backward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target) {
    return pred.array() - target.array();
}
