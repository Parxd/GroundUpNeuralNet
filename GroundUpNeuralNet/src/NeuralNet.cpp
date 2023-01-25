#include <algorithm>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "NeuralNet.h"

NeuralNet::NeuralNet(std::vector<int> struc, double lR) : architecture(struc), eta(lR) {
	for (int i = 1; i < architecture.size(); ++i) {
		const int neuronCount = architecture[i];
		Eigen::VectorXd n(neuronCount);
		Eigen::MatrixXd w(architecture[i - 1], neuronCount);
		Eigen::VectorXd b(neuronCount);
		neuron.emplace_back(n);
		weight.emplace_back(w);
		bias.emplace_back(b);
	}
}

