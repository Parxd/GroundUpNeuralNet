#include "../include/Layer.h"

Layer::Layer(int numInputs, int numOutputs):
    inputFeatures(numInputs), outputFeatures(numOutputs)
{
    activationInputs = Eigen::VectorXf(inputFeatures);
    weights = Eigen::MatrixXf::Random(inputFeatures, outputFeatures);
    bias = Eigen::MatrixXf::Random(1, outputFeatures);
}

void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output)
{

}

void backward(const Eigen::MatrixXf& dout, Eigen::MatrixXf& ddout)
{

}