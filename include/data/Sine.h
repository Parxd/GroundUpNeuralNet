#ifndef GROUNDUPNEURALNET_SINE_H
#define GROUNDUPNEURALNET_SINE_H

#include <Eigen/Dense>

class Sine {
public:
    Sine() = delete;

    ~Sine() = delete;

    /**
     * @brief Noisy sine wave data generator
     * @param num - Number of data points desired (>500,000 for best results)
     * @param tolerance - Amount of noise added to each point
     * @param xStretchFactor - The factor by which x-values are stretched out
     * @param ySquashFactor - The factor by which y-values are squashed (amplitude reduction)
     * @param errorSplit - Percentage of incorrect data points desired from the returned matrix
     * @return - 3-by-num matrix, where each column is an (x, y) coordinate and its label, with num number of columns
     */
    static Eigen::MatrixXf generate(int num, float errorSplit, float tolerance, int xStretchFactor, float ySquashFactor);
};

#endif //GROUNDUPNEURALNET_SINE_H
