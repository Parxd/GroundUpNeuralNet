#ifndef GROUNDUPNEURALNET_SIGMOID_H
#define GROUNDUPNEURALNET_SIGMOID_H

#include <cmath>
#include "../layers/BaseModule.h"

class BaseModule;

class Sigmoid : public BaseModule {
public:
    Sigmoid() = default;

    ~Sigmoid() override = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;

    Eigen::MatrixXf backward(const Eigen::MatrixXf &dLA) override;

    [[nodiscard]] std::string getName() const override;

private:
    Eigen::MatrixXf curActivation;
};

#endif