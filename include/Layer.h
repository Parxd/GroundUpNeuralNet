#ifndef LAYER_H
#define LAYER_H

#include "BaseModule.h"

class Layer : public BaseModule
{
public:
    Layer(int numInputs, int numOutputs);
    ~Layer() = default;

private:
    Eigen::VectorXf activationInputs;
    Eigen::MatrixXf weights;
    Eigen::MatrixXf bias;
    int inputFeatures;
    int outputFeatures;
};

#endif