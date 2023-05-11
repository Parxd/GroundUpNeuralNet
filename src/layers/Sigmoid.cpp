#include <iostream>
#include "../../include/layers/Sigmoid.h"

Eigen::MatrixXf Sigmoid::forward(const Eigen::MatrixXf &input) {
    // Cache computed sigmoid input, rather than raw input (for use during backpropagation)
    curActivation = input.array().unaryExpr(
            [] (float x) -> float {return float(1) / (1 + float(exp(-double(x))));}
            );
    return curActivation;
}

Eigen::MatrixXf Sigmoid::backward(const Eigen::MatrixXf &input) {
    Eigen::MatrixXf output = curActivation.array() * (1 - curActivation.array());
    return output.array() *= input.array();
}

std::string Sigmoid::getName() const {
    return "Sigmoid";
}