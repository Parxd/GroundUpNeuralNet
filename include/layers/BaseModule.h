#ifndef GROUNDUPNEURALNET_BASEMODULE_H
#define GROUNDUPNEURALNET_BASEMODULE_H

#include <memory>
#include "Eigen/Dense"

class BaseModule
{
public:
    template <typename T>
    static std::unique_ptr<BaseModule> make()
    {
        return std::make_unique<T>();
    }

    template <typename T>
    static std::unique_ptr<BaseModule> make(int in, int out)
    {
        return std::make_unique<T>(in, out);
    }

    virtual ~BaseModule() = default;
    virtual void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) = 0;
    virtual void backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output) = 0;
    [[nodiscard]] virtual std::string getName() const = 0;
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