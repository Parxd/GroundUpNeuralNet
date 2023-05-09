#include <iostream>
#include "../../include/layers/Softmax.h"

Eigen::MatrixXf Softmax::forward(const Eigen::MatrixXf &input) {
    Eigen::MatrixXf output = input.array().exp();
    for (long i = 0; i < output.cols(); ++i) {
        output.col(i) /= output.col(i).sum();
    }
    // Cache computed softmax input, rather than raw input (for use during backpropagation)
    curActivation = output;
    return output;
}

Eigen::MatrixXf Softmax::backward(const Eigen::MatrixXf &dLA) {
    Eigen::MatrixXf output = dLA.array() *
            (curActivation.array() * (1 - curActivation.array()).array()).array();
    return output;
}

std::string Softmax::getName() const {
    return "Softmax";
}
