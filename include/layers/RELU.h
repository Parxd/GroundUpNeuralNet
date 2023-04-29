#ifndef GROUNDUPNEURALNET_RELU_H
#define GROUNDUPNEURALNET_RELU_H

#include "BaseModule.h"

// Forward declaration
class BaseModule;

class RELU : public BaseModule
{
public:
    RELU() = default;

    ~RELU() = default;

    /**
     * @brief
     * @param
     * @param
    */
    void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;

    /**
     * @brief
     * @param
     * @param
    */
    void backward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;

    [[nodiscard]] std::string getName() const override;

private:
    Eigen::MatrixXf storedInput;
};

#endif