#ifndef GROUNDUPNEURALNET_SIGMOID_H
#define GROUNDUPNEURALNET_SIGMOID_H

#include <cmath>
#include "../layers/BaseModule.h"

class BaseModule;

class Sigmoid {
public:
    Sigmoid() = default;

    ~Sigmoid() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input);

    Eigen::MatrixXf backward(const Eigen::MatrixXf &dLA);

    [[nodiscard]] std::string getName() const;

private:
    static float sigmoidFunction(float x);

    Eigen::MatrixXf curActivation;
};

#endif