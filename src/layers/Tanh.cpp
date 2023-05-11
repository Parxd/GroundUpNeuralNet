#include "../../include/layers/Tanh.h"

Eigen::MatrixXf Tanh::forward(const Eigen::MatrixXf &input) {
    curActivation = input.array().tanh();
    return curActivation;
}

Eigen::MatrixXf Tanh::backward(const Eigen::MatrixXf &dLA) {
    return (1 - curActivation.array().square()) * curActivation.array();
}

std::string Tanh::getName() const {
    return "Tanh";
}