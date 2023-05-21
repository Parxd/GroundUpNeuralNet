#include "../include/containers/Container.h"
#include "../include/containers/Trainer.h"
#include "../include/data/Sine.h"

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    Container cont(
            new Linear(2, 10),
            new ReLU(),
            new Linear(10, 10),
            new Sigmoid(),
            new Linear(10, 2)
            );
    auto data = Sine::generate(500000, 0.2, 10, 5, 1.1);
    Eigen::MatrixXf features = data.topRows(2);
    Eigen::MatrixXf labels = data.bottomRows(2);
    Trainer<CE>::train(cont, features, labels, 32, 10, 1);
    return 0;
}