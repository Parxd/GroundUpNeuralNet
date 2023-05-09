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
    return curActivation.array().unaryExpr(
            [] (float x) -> float {return float(bool(x > 0) * 1);}
            ) * input.array();
}

std::string RELU::getName() const
{
    return "ReLU";
}
