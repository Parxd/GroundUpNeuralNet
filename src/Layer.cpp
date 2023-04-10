#include "../include/Layer.h"

Layer::Layer(int numInputs, int numOutputs):
    inputFeatures(numInputs), outputFeatures(numOutputs)
{
    weights = Eigen::MatrixXf::Random(inputFeatures, outputFeatures);
    bias = Eigen::MatrixXf::Random(1, outputFeatures);
}

void Layer::forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{
    inputs = input;
    output = input.matrix() * weights.matrix();
    output += bias;
}

void Layer::backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output)
{
    weights = weights - (eta * (inputs.transpose() * dEW));
    bias = bias - (eta * output);
    output = dEW * weights.transpose();
}

void Layer::setLearningRate(const float& learningRate)
{
    eta = learningRate;
}