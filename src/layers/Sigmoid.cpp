#include "../../include/layers/Sigmoid.h"

Eigen::MatrixXf Sigmoid::forward(const Eigen::MatrixXf &input) {
    curActivation = input;
    Eigen::MatrixXf output = input.array().unaryExpr(
            [] (float x) -> float {return sigmoidFunction(x);}
            );
    return std::move(output);
}

Eigen::MatrixXf Sigmoid::backward(const Eigen::MatrixXf &dLA) {
    Eigen::MatrixXf output = curActivation.array().unaryExpr(
            [] (float x) -> float {return sigmoidFunction(x) * (1 - sigmoidFunction(x));}
            );
    output = output.array() * dLA.array();
    return std::move(output);
}

std::string Sigmoid::getName() const {
    return "Sigmoid";
}

float Sigmoid::sigmoidFunction(float x) {
    return float(1) / (1 + float(exp(-double(x))));
}
