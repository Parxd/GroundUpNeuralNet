#ifndef GROUNDUPNEURALNET_CONTAINER_H
#define GROUNDUPNEURALNET_CONTAINER_H

#include <memory>
#include <iostream>
#include <vector>
#include "../layers/BaseModule.h"
#include "../layers/Linear.h"
#include "../layers/ReLU.h"
#include "../layers/LeakyReLU.h"
#include "../layers/Sigmoid.h"
#include "../layers/Softmax.h"
#include "../../include/losses/MSE.h"
#include "../../include/losses/CE.h"

class Container
{
public:
    Container() = default;

    ~Container() = default;

    /*
     * @brief PREFERRED variadic constructor - accepts std::unique_ptr from BaseModule layer factories
     * @param layer - Variable number of std::unique_ptr of BaseModule type
     * @return Container object with layers added to internal vector
     */
    template<typename... T>
    explicit Container(std::unique_ptr<T>&&... layer)
    {
        (mLayers.push_back(std::move(layer)), ...);
    }

    /*
     * @brief Variadic constructor - converts raw base pointers into std::unique_ptr for trivial destruction
     * @param layer - Variable number of RAW BaseModule pointers
     * @return Container object with layers added to internal vector
     */
    template<typename... T>
    explicit Container(T*... layer)
    {
        (mLayers.push_back(std::unique_ptr<BaseModule>(layer)), ...);
    }

    /**
     * @brief Feedforward method
     * @param input - Data to be fed through network
     * @return Network's output values/probabilities
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);

    /**
     * @brief Backpropagation method
     * @param errorDerivative - The derivative of error with respect to network's output activations (takes in the
     *                          matrix from LossFunction::backward(modelOutput, target), see documentation for more)
     */
    void backward(const Eigen::MatrixXf& errorDerivative);

    /*
     * @brief Prints order and description of each layer in internal vector
     */
    void view();

    /*
     * @brief Saves model's weights & biases to directory
     * @param directory - File to save to (.CSV)
     * @param name - Model name to be saved under
     */
    void save(const std::string& file, const std::string& name = "");

    /**
     * @brief Loads model's weights & biases from directory
     * @param file - File to load from (.CSV)
     */
    void load(const std::string& file);

private:
    std::vector<std::unique_ptr<BaseModule>> mLayers;

    void write(const std::string& file, const std::string& name);
};

#endif //GROUNDUPNEURALNET_CONTAINER_H
