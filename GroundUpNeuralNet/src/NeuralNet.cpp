#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <Eigen/Dense>
#include "NeuralNet.h"

class MathFunctor {
public:
	MathFunctor(int select) : option(select) {}
	float operator() (float x) const {
		if (option == 1) {
			return 1 / (1 + std::exp(-x));
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

void NeuralNet::printNeurons() {
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

auto NeuralNet::ForwardProp(Eigen::VectorXf mat) {
	activation[0] = mat;
	Eigen::VectorXf prev = mat;
	for (int i = 1; i < architecture.size(); ++i) {
		Eigen::VectorXf y = weight[i] * prev;
		y += bias[i];
		y = y.unaryExpr(MathFunctor(1));
		activation[i] = y;
		prev = y;
	}
	return prev;
}