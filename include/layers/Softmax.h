#ifndef GROUNDUPNEURALNET_SOFTMAX_H
#define GROUNDUPNEURALNET_SOFTMAX_H

#include "../layers/BaseModule.h"

class BaseModule;

class Softmax : public BaseModule {
public:
    Softmax() = default;

    ~Softmax() override = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;

    Eigen::MatrixXf backward(const Eigen::MatrixXf &dLA) override;

    [[nodiscard]] std::string getName() const override;

private:
    Eigen::MatrixXf curActivation;
};

#endif // GROUNDUPNEURALNET_SOFTMAX_H
