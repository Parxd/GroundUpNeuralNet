#pragma once
#include <iostream>
#include <Eigen/Dense>

class NeuralNet
{
private:
	Eigen::MatrixXd a;
	Eigen::VectorXd w;
	Eigen::VectorXd b;

public:
	NeuralNet();

	NeuralNet(Eigen::VectorXd input);

	void FeedForward();

	void BackProp();

	Eigen::VectorXd MiniBatch();

	double Sigmoid(int x);

	double SigmoidPrime(int x);

	double ReLU(int x);

	double ReLUPrime(int x);
};

