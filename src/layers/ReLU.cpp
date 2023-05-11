#include <iostream>
#include "../../include/layers/ReLU.h"

Eigen::MatrixXf ReLU::forward(const Eigen::MatrixXf& input)
{
    curActivation = input.array().unaryExpr(
            [] (float x) -> float {return std::max(0.0f, x);}
            );
    return curActivation;
}

Eigen::MatrixXf ReLU::backward(const Eigen::MatrixXf& input)
{
    Eigen::MatrixXf output = curActivation.array().unaryExpr(
            [] (float x) -> float {return float(bool(x > 0) * 1);}
            );
    return output.array() *= input.array();
}

std::string ReLU::getName() const
{
    return "ReLU";
}
