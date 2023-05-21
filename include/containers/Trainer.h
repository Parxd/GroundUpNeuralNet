#ifndef GROUNDUPNEURALNET_TRAINER_H
#define GROUNDUPNEURALNET_TRAINER_H

#include "../containers/Container.h"

class Trainer {
public:
    Trainer() = delete;

    ~Trainer() = delete;

    static void train(Container& model,
                      Eigen::MatrixXf& trainData,
                      Eigen::MatrixXf& trainLabels,
                      int batchSize,
                      int epochs,
                      int printFreq);
};

#endif //GROUNDUPNEURALNET_TRAINER_H
