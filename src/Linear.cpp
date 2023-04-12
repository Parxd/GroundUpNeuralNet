#include "../include/Linear.h"

Linear::Linear(int numInputs, int numOutputs):
    inputFeatures(numInputs), outputFeatures(numOutputs), eta()
{
    weights = Eigen::MatrixXf::Random(inputFeatures, outputFeatures);
    bias = Eigen::MatrixXf::Random(1, outputFeatures);
}

void Linear::forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{
    storedInput = input;
    output = input.matrix() * weights.matrix();
    output += bias;
}

void Linear::backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output)
{
    weights = weights - (eta * (storedInput.transpose() * dEW));
    bias = bias - (eta * output);
    output = dEW * weights.transpose();
}

void Linear::setLearningRate(const float& learningRate)
{
    eta = learningRate;
}

void Linear::setWeight(const Eigen::MatrixXf& newWeights)
{
    weights = newWeights;
}

void Linear::setBias(const Eigen::MatrixXf& newBias)
{
    bias = newBias;
}

const Eigen::MatrixXf& Linear::getWeight() const
{
    return weights;
}

const Eigen::MatrixXf& Linear::getBias() const
{
    return bias;
}

void Linear::accept(Visitor& visitor)
{
    visitor.visit(*this);
}

std::string Linear::getName()
{
    return name;
}