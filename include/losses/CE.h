#ifndef GROUNDUPNEURALNET_CE_H
#define GROUNDUPNEURALNET_CE_H

#include <Eigen/Dense>

struct CE {
    CE() = delete;

    ~CE() = delete;

    static float forward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target);

    static Eigen::MatrixXf backward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target);
};

#endif // GROUNDUPNEURALNET_CE_H
