#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>

class NeuralNet
{
public:
	explicit NeuralNet(std::vector<int> struc, double lR, int function);
	void printActivations();
	void printWeights();
	void printBias();
	float Sigmoid(float x) const;
	auto ForwardProp(const Eigen::VectorXf& mat);
	void BackwardProp(const Eigen::VectorXf& start);
	Eigen::VectorXf CostFunc(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted);

private:
	int activationType;
	double eta;
	std::vector<Eigen::VectorXf> activation;
	std::vector<Eigen::MatrixXf> weight;
	std::vector<Eigen::VectorXf> bias;
	std::vector<int> architecture;
};
