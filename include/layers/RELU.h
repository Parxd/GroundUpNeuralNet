#ifndef GROUNDUPNEURALNET_RELU_H
#define GROUNDUPNEURALNET_RELU_H

#include "BaseModule.h"

// Forward declaration
class BaseModule;

class RELU : public BaseModule
{
public:
    RELU() = default;

    ~RELU() override = default;

    /**
     * @brief Feedforward previous layer through ReLU activation function
     * @param input - The input matrix that is fed into this layer from the previous layer
     * @param output - Resulting matrix after ReLU function applied
    */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    /**
     * @brief
     * @param
     * @param
    */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& input) override;

    [[nodiscard]] std::string getName() const override;

private:    
    Eigen::MatrixXf curActivation;
};

#endif