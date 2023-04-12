#ifndef RELU_H
#define RELU_H

#include "BaseModule.h"
#include "Visitor.h"

// Forward declaration
class BaseModule;

class RELU : public BaseModule
{
public:
    RELU() = default;

    ~RELU() = default;

    void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;

    void backward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;

    /**
     * @brief Accept the Visitor class
     * @param visitor - Reference to Visitor object to be accepted
    */
    void accept(Visitor& visitor);

    std::string getName() override;

private:
    std::string name = "RELU";
    Eigen::MatrixXf storedInput;
};

#endif