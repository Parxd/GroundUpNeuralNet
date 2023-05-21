#include "../../include/containers/Trainer.h"

void
Trainer::train(Container &model, Eigen::MatrixXf &trainData, Eigen::MatrixXf &trainLabels, int batchSize, int epochs, int printFreq) {
    int numBatches = int(trainData.cols()) / batchSize;
    for (int i = 0; i < epochs; ++i) {
        float loss = 0.f;
        for (int j = 0; j < numBatches; ++j) {
            auto batchData = trainData.block(0, j * batchSize, trainData.rows(), batchSize);
            auto batchLabels = trainLabels.block(0, j * batchSize, trainData.rows(), batchSize);
            auto forward = model.forward(batchData);
            loss += MSE::forward(forward, batchLabels);
            model.backward(MSE::backward(forward, batchLabels));
        }
        loss /= float(numBatches);
        std::cout << loss;
    }
}
