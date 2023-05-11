#include <iostream>
#include "../../include/layers/Softmax.h"

Eigen::MatrixXf Softmax::forward(const Eigen::MatrixXf &input) {
    // Cache computed softmax input, rather than raw input (for use during backpropagation)
    curActivation = input.array().exp();
    for (long i = 0; i < curActivation.cols(); ++i) {
        curActivation.col(i) /= curActivation.col(i).sum();
    }
    return curActivation;
}

Eigen::MatrixXf Softmax::backward(const Eigen::MatrixXf &dLA) {
    Eigen::MatrixXf output = curActivation.array() * (1 - curActivation.array()).array();
    return output.array() *= dLA.array();
}

std::string Softmax::getName() const {
    return "Softmax";
}
