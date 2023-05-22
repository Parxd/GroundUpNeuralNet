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
                std::seed_seq seed{r()};
                std::mt19937 eng(seed);
                Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permX(trainData.cols());
                permX.setIdentity();
                std::shuffle(permX.indices().data(), permX.indices().data()+permX.indices().size(), eng);
                trainData = trainData * permX;
                trainLabels = trainLabels * permX;
            }
            float trainAccuracy = 0.f;
            float loss = 0.f;
            for (int j = 0; j < numBatches; ++j) {
                auto batchData = trainData.block(0, j * batchSize, trainData.rows(), batchSize);
                auto batchLabels = trainLabels.block(0, j * batchSize, trainData.rows(), batchSize);
                auto forward = model.forward(batchData);
                trainAccuracy += evaluate(forward, batchLabels) / float(batchSize);
                loss += LossFunction::forward(forward, batchLabels);
                model.backward(LossFunction::backward(forward, batchLabels));
            }
            loss /= float(numBatches);
            trainAccuracy /= float(numBatches);
            if (i % printFreq == 0) {
                std::cout << "Epoch " << i + 1 << ": Loss: " << loss << "; Training Accuracy: " << trainAccuracy << std::endl;
            }
        }
    }
private:
    static inline float evaluate(const Eigen::MatrixXf& res, const Eigen::MatrixXf& target) {
        float accuracy = 0;
        for (int i = 0; i < target.cols(); ++i) {
            int argmaxTarget = -1;
            float maxTarget = target.col(i).maxCoeff();
            for (int j = 0; j < target.rows(); ++j) {
                if (target.col(i).row(j).data()[0] == maxTarget) {
                    argmaxTarget = j;
                }
            }
            int argmaxRes = 0;
            float maxRes = res.col(i).maxCoeff();
            for (int k = 0; k < res.rows(); ++k) {
                if (res.col(i).row(k).data()[0] == maxRes) {
                    argmaxRes = k;
                }
            }
            if (argmaxTarget == argmaxRes) {
                ++accuracy;
            }
        }
        return accuracy;
    }
};

#endif //GROUNDUPNEURALNET_TRAINER_H
