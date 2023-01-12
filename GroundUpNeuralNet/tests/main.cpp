#include "..\src\NeuralNet.h"
#include <Eigen/Dense>
#include <iostream>

int main()
{
	std::vector<Eigen::MatrixXd> test;
	Eigen::Matrix<double, 4, 3> m1;

	m1 << 1.5, 2.5, 3.5,
		4.5, 5.5, 6.5,
		7.5, 8.5, 9.5,
		10.5, 11.5, 12.5;
	std::cout << m1;
	test.push_back(m1);
}