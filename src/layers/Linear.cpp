#include "../../include/layers/Linear.h"
#include <iostream>

Linear::Linear(int numInputs, int numOutputs):
    inputFeatures(numInputs), outputFeatures(numOutputs)
{
    weights = Eigen::MatrixXf::Random(outputFeatures, inputFeatures);
    bias = Eigen::VectorXf::Random(outputFeatures, 1);
}

Eigen::MatrixXf Linear::forward(const Eigen::MatrixXf& input)
{
    curActivation = input;
    Eigen::MatrixXf nextActivation = weights * input;
    for (long i = 0; i < nextActivation.cols(); ++i)
    {
        nextActivation.col(i) += bias;
    }
    return nextActivation;
}

Eigen::MatrixXf Linear::backward(const Eigen::MatrixXf& dLA)
{
    // One-time in-place transpose of curActivation for matrix multiplication
    if (curActivation.cols() != inputFeatures) {
        curActivation.transposeInPlace();
    }
    // Gradient descent - updating weights/biases
    auto gradient = dLA * curActivation;
    weights = weights.array() - eta * gradient.array();
    bias = bias.array() - (eta * dLA.rowwise().mean()).array();
    Eigen::MatrixXf output = dLA.transpose() * weights;
    output.transposeInPlace();
    return output;
}

std::string Linear::getName() const
{
    return "Linear";
}

int Linear::getInputs() const
{
    return inputFeatures;
}

int Linear::getOutputs() const
{
    return outputFeatures;
}

float Linear::getLR() const
{
    return eta;
}

void Linear::setLearningRate(const float& learningRate)
{
    if (learningRate >= 1)
    {
        std::cerr << "Warning: Learning rate greater than 1 may have unwanted effects.";
    }
    eta = learningRate;
}

void Linear::setWeight(Eigen::MatrixXf& newWeights)
{
    assert(
            newWeights.cols() == inputFeatures && newWeights.rows() == outputFeatures ||
            newWeights.cols() == outputFeatures && newWeights.rows() == inputFeatures
            );
    if (newWeights.cols() == outputFeatures) {
        newWeights.transposeInPlace();
    }
    weights = newWeights;
}

void Linear::setBias(Eigen::MatrixXf& newBias)
{
    assert(newBias.rows() == outputFeatures);
    bias = newBias;
}

const Eigen::MatrixXf& Linear::getWeight() const
{
    return weights;
}

const Eigen::VectorXf& Linear::getBias() const
{
    return bias;
}