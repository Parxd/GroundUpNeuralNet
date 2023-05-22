#ifndef GROUNDUPNEURALNET_RELU_H
#define GROUNDUPNEURALNET_RELU_H

#include "BaseModule.h"

// Forward declaration
class BaseModule;

class ReLU : public BaseModule
{
public:
    ReLU() = default;

    ~ReLU() override = default;

    /**
     * @brief Feedforward previous layer through ReLU activation function
     * @param input - The input matrix that is fed into this layer from the previous layer
     * @return Resulting matrix after ReLU function applied
    */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    /**
     * @brief ReLU derivative
     * @param input - The propagated error from higher layers
     * @return Error w/ respect to ReLU activation
    */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& input) override;

    [[nodiscard]] std::string getName() const override;

private:    
    Eigen::MatrixXf curActivation;
};

#endif