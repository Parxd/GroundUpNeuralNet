#ifndef BASEMODULE_H
#define BASEMODULE_H

#include <Eigen/Dense>

class BaseModule
{
public:
    virtual ~BaseModule() = default;
    virtual void forward(const Eigen::MatrixXf& intput, Eigen::MatrixXf& output) = 0;
    virtual void backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output) = 0;
    [[nodiscard]] virtual std::string getName() = 0;
};

#endif