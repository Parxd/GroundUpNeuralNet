#include "../../include/layers/LeakyReLU.h"

Eigen::MatrixXf LeakyReLU::forward(const Eigen::MatrixXf &input) {
    curActivation = input.array().unaryExpr(
            [] (float x) -> float {return std::max(0.2f, x);}
    );
    return curActivation;
}

Eigen::MatrixXf LeakyReLU::backward(const Eigen::MatrixXf &dLA) {
    Eigen::MatrixXf output = curActivation.array().unaryExpr(
            [] (float x) -> float {return (x >= 0) ? 1 : 0.2f;}
    );
    return output.array() *= dLA.array();
}

std::string LeakyReLU::getName() const {
    return "LeakyReLU";
}
