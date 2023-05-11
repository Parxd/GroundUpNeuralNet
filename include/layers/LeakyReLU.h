#ifndef GROUNDUPNEURALNET_LEAKYRELU_H
#define GROUNDUPNEURALNET_LEAKYRELU_H

#include "../layers/BaseModule.h"

class LeakyReLU : public BaseModule {
public:
    LeakyReLU() = default;

    ~LeakyReLU() override = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;

    Eigen::MatrixXf backward(const Eigen::MatrixXf &dLA) override;

    [[nodiscard]] std::string getName() const override;

private:
    Eigen::MatrixXf curActivation;
};

#endif //GROUNDUPNEURALNET_LEAKYRELU_H
