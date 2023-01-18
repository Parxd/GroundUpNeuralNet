#include <algorithm>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "NeuralNet.h"

NeuralNet::NeuralNet(std::vector<int> struc) : architecture(struc) {
	for (int i = 1; i < architecture.size(); ++i) {
		matrixPtr layerW(new Eigen::MatrixXd);
		matrixPtr layerB(new Eigen::MatrixXd);
		weight.push_back(layerW);
		bias.push_back(layerB);
	}
}

