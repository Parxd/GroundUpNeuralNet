#ifndef GROUNDUPNEURALNET_CONTAINER_H
#define GROUNDUPNEURALNET_CONTAINER_H

#include <memory>
#include <iostream>
#include <vector>
#include "../layers/BaseModule.h"
#include "../layers/Linear.h"
#include "../layers/ReLU.h"
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

    /*
     * @brief Prints order and description of each layer in internal vector
     */
    void view();

    /**
     * @brief Feedforward method
     * @param input - Data to be fed through network
     * @return Network's output values/probabilities
     */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);

    /**
     * @brief Backpropagation method (defined here due to use of template)
     * @tparam Loss - Type of loss function used
     * @param pred - Network's output values/probabilities
     * @param target - Correct label (target) values/probabilities
     * @return Error metric calculated by loss function after feedforward
     */
    template <typename Loss>
    float backward(const Eigen::MatrixXf& pred, const Eigen::MatrixXf& target) {
        if (!(dynamic_cast<Softmax *>(mLayers.back().get())) && std::is_same<Loss, CE>::value) {
            std::cerr << "Using cross-entropy loss with a final non-softmax layer is not recommended "
                         "for multi-classification.";
        }
        float loss = Loss::forward(pred, target);
        auto errorDerivative = Loss::backward(pred, target);

        auto output = mLayers.back()->backward(errorDerivative);
        for (auto it = mLayers.rbegin() + 1; it != mLayers.rend(); ++it) {
            output = (*it)->backward(output);
        }
        return loss;
    }

private:
    std::vector<std::unique_ptr<BaseModule>> mLayers;
};

#endif //GROUNDUPNEURALNET_CONTAINER_H
