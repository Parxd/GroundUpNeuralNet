#include "../../include/layers/Linear.h"
#include <iostream>

Linear::Linear(int numInputs, int numOutputs):
    inputFeatures(numInputs), outputFeatures(numOutputs)
{
    weights = Eigen::MatrixXf::Random(outputFeatures, inputFeatures);
    bias = Eigen::VectorXf::Random(outputFeatures, 1);
}

Linear::Linear(int numInputs, int numOutputs, float lR):
    inputFeatures(numInputs), outputFeatures(numOutputs), eta(lR)
{
    weights = Eigen::MatrixXf::Random(outputFeatures, inputFeatures);
    bias = Eigen::VectorXf::Random(outputFeatures, 1);
}

void Linear::forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{
    storedInput = input;
    output = weights * input;
    for (long i = 0; i < output.rows(); ++i) {
        output.row(i).array() += bias.array();
    }
}

void Linear::backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output)
{
    weights = weights.array() - (eta * (storedInput.transpose() * dEW).array());
    bias = bias.array() - (eta * dEW.colwise().mean().array());
    // TODO: Fix this next line
    output = dEW * weights;
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
