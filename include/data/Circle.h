#ifndef GROUNDUPNEURALNET_CIRCLE_H
#define GROUNDUPNEURALNET_CIRCLE_H

#include <Eigen/Dense>

class Circle {
public:
    Circle() = delete;

    ~Circle() = delete;

    static Eigen::MatrixXf generate();
};

#endif //GROUNDUPNEURALNET_CIRCLE_H
