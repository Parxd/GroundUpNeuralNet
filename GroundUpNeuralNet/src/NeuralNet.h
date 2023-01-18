#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Dense>

typedef std::unique_ptr<Eigen::MatrixXd> matrixPtr;

class NeuralNet
{
private:
	std::vector<matrixPtr> weight;
	std::vector<matrixPtr> bias;
	std::vector<int> architecture;
	
public:
	NeuralNet(std::vector<int> struc);
};
