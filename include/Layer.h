#ifndef LAYER_H
#define LAYER_H

#include "BaseModule.h"

class Layer : public BaseModule
{
public:
    Layer(int numInputs, int numOutputs);
    ~Layer() = default;
    /**
     * @brief Feedforward method of a linear layer class
     * @param [input] The matrix that is fed into this layer from the previous layer 
     * @param [output] 
    */
    void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;
    void backward(const Eigen::MatrixXf& dout, Eigen::MatrixXf& ddout) override;

private:
    Eigen::VectorXf activationInputs;
    Eigen::MatrixXf weights;
    Eigen::MatrixXf bias;
    int inputFeatures;
    int outputFeatures;
};

#endif