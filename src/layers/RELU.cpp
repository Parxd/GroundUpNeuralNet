#include "../../include/layers/RELU.h"

void RELU::forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{
    curActivation = input;
    output = curActivation.array().unaryExpr(
            [] (float x) -> float {return std::max(0.0f, x);}
            );
}

void RELU::backward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{
    output = curActivation.array().unaryExpr(
            [] (float x) -> float {return float(bool(x > 0) * 1);}
            );
    output = output.array() * input.array();
}

std::string RELU::getName() const
{
    return "ReLU";
}
