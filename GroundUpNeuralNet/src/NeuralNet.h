#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>

class NeuralNet
{
private:
	std::vector<int> struc;
	std::vector<Eigen::MatrixXd> w;
	Eigen::MatrixXd b;
	
public:
	NeuralNet();

	NeuralNet(std::vector<int> struc);

	void StrucChange(std::vector<int> newStruc);

	void FeedForward();

	void BackProp();

	Eigen::VectorXd MiniBatch(Eigen::VectorXd batch);

	double Sigmoid(int x);

	double SigmoidPrime(int x);

	double ReLU(int x);

	double ReLUPrime(int x);
};
