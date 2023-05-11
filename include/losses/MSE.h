#ifndef GNN_MSE_HPP
#define GNN_MSE_HPP

#include <Eigen/Dense>

struct MSE {
    MSE() = delete;

    ~MSE() = delete;

    static float forward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target);

    static Eigen::MatrixXf backward(const Eigen::MatrixXf &pred, const Eigen::MatrixXf &target);
};

#endif //GNN_MSE_HPP
