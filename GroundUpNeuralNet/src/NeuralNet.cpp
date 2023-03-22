#include <algorithm>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "NeuralNet.h"

class ActivationLoss {
public:
	ActivationLoss(int select) : option(select) {}

	float Sigmoid(float x) const {
		return 1 / (1 + std::exp(-x));
	}

	float SigmoidPrime(float x) const {
		return Sigmoid(x) * (1 - Sigmoid(x));
	}

	float operator() (float x) const {
		if (option == 1) {
			return Sigmoid(x);
		}
		else if (option == 2) {
			return SigmoidPrime(x);
		}
	}
private:
	int option;
};

NeuralNet::NeuralNet(std::vector<int> struc, double lR = 0.001, int function = 0) : architecture(struc), eta(lR), activationType(function) {
	if (activationType < 0 || activationType > 2) {
		throw std::out_of_range("Invalid activation function. Valid values are [0 - 2]");
	}
	for (int i = 1; i < architecture.size(); ++i) {
		const int neuronCount = architecture[i];
		Eigen::VectorXf n(neuronCount);
		Eigen::MatrixXf w(neuronCount, architecture[i - 1]);
		Eigen::VectorXf b(neuronCount);
		activation.emplace_back(n.setRandom());
		weight.emplace_back(w.setRandom());
		bias.emplace_back(b.setRandom());
	}
}

void NeuralNet::printActivations() {
	for (auto& i : activation) {
		std::cout << "newL" << std::endl;
		std::cout << i << std::endl;
	}
}

void NeuralNet::printWeights() {
	for (auto& i : weight) {
		std::cout << "newL" << std::endl;
		std::cout << i << std::endl;
	}
}

void NeuralNet::printBias() {
	for (auto& i : bias) {
		std::cout << "newL" << std::endl;
		std::cout << i << std::endl;
	}
}

auto NeuralNet::ForwardProp(const Eigen::VectorXf& mat) {
	Eigen::VectorXf prev = mat;
	activation[0] = prev;
	for (int i = 1; i < architecture.size(); ++i) {
		Eigen::VectorXf y = weight[i] * prev;
		y += bias[i];
		y = y.unaryExpr(ActivationLoss(1));
		activation[i] = y;
		prev = y;
	}
	return prev;
}

void NeuralNet::BackwardProp(const Eigen::VectorXf& actual) {
	Eigen::VectorXf y = actual;
	Eigen::VectorXf y_hat = activation.back();
	Eigen::VectorXf error = CostFunc(y, y_hat);
	
}

Eigen::VectorXf NeuralNet::CostFunc(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
{
	return actual - predicted;
}
