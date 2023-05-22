#include <fstream>
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
            new Linear(10, 2),
            new Softmax()
            );
    auto data = Sine::generate(1000000, 0.2, 200, 50, 1.1);
    Eigen::MatrixXf features = data.topRows(2);
    Eigen::MatrixXf labels = data.bottomRows(2);
    Trainer<MSE>::train(cont, features, labels, 5, 10, 1, true);
    cont.save("../src/model.csv");

    return 0;
}