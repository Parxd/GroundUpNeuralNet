#ifndef GROUNDUPNEURALNET_TANH_H
#define GROUNDUPNEURALNET_TANH_H

#include "../../include/layers/BaseModule.h"

class Tanh : public BaseModule {
public:
    Tanh() = default;

    ~Tanh() override = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;

    Eigen::MatrixXf backward(const Eigen::MatrixXf &dLA) override;

    std::string getName() const override;

private:
    Eigen::MatrixXf curActivation;
};

#endif //GROUNDUPNEURALNET_TANH_H
