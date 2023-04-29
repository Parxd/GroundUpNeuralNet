#ifndef GROUNDUPNEURALNET_BASEMODULE_H
#define GROUNDUPNEURALNET_BASEMODULE_H

#include <memory>
#include <Eigen/Dense>

class BaseModule
{
public:
    virtual ~BaseModule() = default;
    virtual void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) = 0;
    virtual void backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output) = 0;
    virtual const std::string getName() const = 0;
    virtual const int getInputs() const
    {
        return 0;
    }
    virtual const int getOutputs() const
    {
        return 0;
    }
    virtual const float getLR() const
    {
        return 0.0f;
    }
};

#endif