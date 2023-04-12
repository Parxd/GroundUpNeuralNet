#ifndef RELU_H
#define RELU_H

#include "../BaseModule.h"

// Forward declaration
class BaseModule;

class RELU : public BaseModule
{
public:
    RELU() = default;

    ~RELU() = default;

    void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;

    void backward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;

private:
    Eigen::MatrixXf storedInput;
};

#endif