#ifndef GROUNDUPNEURALNET_TRAINER_H
#define GROUNDUPNEURALNET_TRAINER_H

#include <random>
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
                      int printFreq,
                      bool shuffle) {
        int numBatches = int(trainData.cols()) / batchSize;
        for (int i = 0; i < epochs; ++i) {
            if (shuffle) {
                std::random_device r;
                std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
                std::mt19937 eng(seed);
                Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permX(trainData.cols());
                permX.setIdentity();
                std::shuffle(permX.indices().data(), permX.indices().data()+permX.indices().size(), eng);
                trainData = trainData * permX;
                trainLabels = trainLabels * permX;
            }
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
