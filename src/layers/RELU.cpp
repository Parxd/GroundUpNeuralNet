#include <iostream>
#include "../../include/layers/RELU.h"

Eigen::MatrixXf RELU::forward(const Eigen::MatrixXf& input)
{
    curActivation = input;
    return curActivation.array().unaryExpr(
            [] (float x) -> float {return std::max(0.0f, x);}
            );
}

Eigen::MatrixXf RELU::backward(const Eigen::MatrixXf& input)
{
    Eigen::MatrixXf output = curActivation.array().unaryExpr(
            [] (float x) -> float {return float(bool(x > 0) * 1);}
            );
    return output.array() *= input.array();
}

std::string RELU::getName() const
{
    return "ReLU";
}
