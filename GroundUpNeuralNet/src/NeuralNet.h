#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Dense>

class NeuralNet
{
private:
	double eta;
	std::vector<Eigen::VectorXd> neuron;
	std::vector<Eigen::MatrixXd> weight;
	std::vector<Eigen::VectorXd> bias;
	std::vector<Eigen::MatrixXd> error;
	std::vector<int> architecture;
	
public:
	NeuralNet(std::vector<int> struc, double lR);
};
