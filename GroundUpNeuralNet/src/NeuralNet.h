#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>

class NeuralNet
{
public:
	explicit NeuralNet(std::vector<int> struc, double lR, int function);
	void printNeurons();
	void printWeights();
	void printBias();
	auto ForwardProp(Eigen::MatrixXf mat);

private:
	int activationType;
	double eta;
	std::vector<Eigen::VectorXf> activation;
	std::vector<Eigen::MatrixXf> weight;
	std::vector<Eigen::VectorXf> bias;
	std::vector<int> architecture;
};
