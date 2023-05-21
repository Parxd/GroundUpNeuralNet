#include "../include/containers/Container.h"
#include "../include/containers/Trainer.h"
#include "../include/data/Sine.h"

int main(int argc, char** argv)
{
    Container cont(
            new Linear(2, 10),
            new ReLU(),
            new Linear(10, 10),
            new ReLU(),
            new Linear(10, 2),
            new Sigmoid()
            );
    auto data = Sine::generate(500000, 0.2, 10, 5, 1.1);
    Eigen::MatrixXf features = data.topRows(2);
    Eigen::MatrixXf labels = data.bottomRows(2);
    // In labels: top row 1 means that it is a correct sine point, bottom row 1 means
    //            that it is an incorrect sine point
    Trainer::train(cont, features, labels, 32, 10, 1);
    return 0;
}