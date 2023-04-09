#include "../include/Layer.h"

Layer::Layer(int numInputs, int numOutputs):
    inputFeatures(numInputs), outputFeatures(numOutputs)
{
    activationInputs = Eigen::VectorXf(inputFeatures);
    weights = Eigen::MatrixXf::Random(inputFeatures, outputFeatures);
    bias = Eigen::MatrixXf::Random(1, outputFeatures);
}

void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& target)
{

}

void backward(Eigen::MatrixXf& ddout, const Eigen::MatrixXf& dout)
{
    
}