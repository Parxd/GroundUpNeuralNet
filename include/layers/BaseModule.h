#ifndef GROUNDUPNEURALNET_BASEMODULE_H
#define GROUNDUPNEURALNET_BASEMODULE_H

#include <memory>
#include "Eigen/Dense"

class BaseModule
{
public:
    virtual ~BaseModule() = default;
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& dLA) = 0;
    [[nodiscard]] virtual std::string getName() const = 0;
    /**
     * @brief Static factory of any NON-linear layer (any activation function layer)
     * @tparam T - Type of activation function layer
     * @return std::unique_ptr to layer
     */
    template <typename T>
    static std::unique_ptr<BaseModule> make()
    {
        return std::make_unique<T>();
    }
    /**
     * @brief Static factory of Linear layer
     * @tparam T - Should ALWAYS be Linear
     * @param in - Number of input features
     * @param out - Number of output features
     * @return std::unique_ptr to new Linear layer
     */
    template <typename T>
    static std::unique_ptr<BaseModule> make(int in, int out)
    {
        return std::make_unique<T>(in, out);
    }
    [[nodiscard]] virtual int getInputs() const
    {
        return 0;
    }
    [[nodiscard]] virtual int getOutputs() const
    {
        return 0;
    }
    [[nodiscard]] virtual float getLR() const
    {
        return 0.0f;
    }
};

#endif