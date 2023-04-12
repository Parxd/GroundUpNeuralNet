#include "../include/RELU.h"

void RELU::forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{
    storedInput = input;
    output = storedInput.array().unaryExpr([] (float x) -> float {return std::max(0.0f, x); });
}

void RELU::backward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{
    output = input.array().unaryExpr([] (float x) -> float {return float(bool(x > 0) * 1); });
}

void RELU::accept(Visitor& visitor)
{
    visitor.visit(*this);
}

std::string RELU::getName()
{
    return name;
}