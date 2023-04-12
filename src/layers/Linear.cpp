#include "../include/layers/Linear.h"
#include <iostream>

Linear::Linear(int numInputs, int numOutputs, float lR):
    inputFeatures(numInputs), outputFeatures(numOutputs), eta(lR)
{
    weights = Eigen::MatrixXf::Random(outputFeatures, inputFeatures);
    bias = Eigen::MatrixXf::Random(outputFeatures, 1);
}

void Linear::forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{
    assert(input.cols() == 1);
    storedInput = input;
    output = weights * input + bias;
}

void Linear::backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output)
{
    weights = weights - (eta * (storedInput.transpose() * dEW));
    bias = bias - (eta * output);
    output = dEW * weights.transpose();
}

const std::string Linear::getName() const
{
    return "Linear";
}

const int Linear::getInputs() const
{
    return inputFeatures;
}

const int Linear::getOutputs() const
{
    return outputFeatures;
}

const float Linear::getLR() const
{
    return eta;
}

void Linear::setLearningRate(const float& learningRate)
{
    eta = learningRate;
}

void Linear::setWeight(const Eigen::MatrixXf& newWeights)
{
    assert(newWeights.rows() ==  outputFeatures && newWeights.cols() == inputFeatures);
    weights = newWeights;
}

void Linear::setBias(const Eigen::MatrixXf& newBias)
{
    assert(bias.rows() == outputFeatures);
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