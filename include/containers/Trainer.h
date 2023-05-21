#ifndef GROUNDUPNEURALNET_TRAINER_H
#define GROUNDUPNEURALNET_TRAINER_H

#include "../containers/Container.h"

template <typename LossFunction>
class Trainer {
public:
    Trainer() = delete;

    ~Trainer() = delete;

    static inline void train(Container& model,
                      Eigen::MatrixXf& trainData,
                      Eigen::MatrixXf& trainLabels,
                      int batchSize,
                      int epochs,
                      int printFreq) {

        int numBatches = int(trainData.cols()) / batchSize;
        for (int i = 0; i < epochs; ++i) {
            float loss = 0.f;
            for (int j = 0; j < numBatches; ++j) {
                auto batchData = trainData.block(0, j * batchSize, trainData.rows(), batchSize);
                auto batchLabels = trainLabels.block(0, j * batchSize, trainData.rows(), batchSize);
                auto forward = model.forward(batchData);
                loss += LossFunction::forward(forward, batchLabels);
                model.backward(LossFunction::backward(forward, batchLabels));
            }
            loss /= float(numBatches);
            if (i % printFreq == 0) {
                std::cout << "Epoch " << i + 1 << " loss: " << loss << std::endl;
            }
        }
    }
};

#endif //GROUNDUPNEURALNET_TRAINER_H
