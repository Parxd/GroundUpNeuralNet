#ifndef BASEMODULE_H
#define BASEMODULE_H

#include <Eigen/Dense>

class BaseModule
{
public:
    virtual ~BaseModule() = default;
    virtual void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& target) = 0;
    virtual void backward(Eigen::MatrixXf& ddOut, const Eigen::MatrixXf& dOut) = 0;
    virtual void print() = 0;
};

#endif