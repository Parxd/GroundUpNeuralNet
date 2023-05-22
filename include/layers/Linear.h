#ifndef GROUNDUPNEURALNET_LINEAR_H
#define GROUNDUPNEURALNET_LINEAR_H

#include "BaseModule.h"

// Forward declaration
class BaseModule;

class Linear : public BaseModule
{
public:
    /**
     * @brief Main constructor of an individual fully-connected layer class
     * @param numInputs - The number of incoming neuron activations from the previous layer
     * @param numOutputs - The number of outgoing neuron activations--also the number of nodes of this layer
    */
    Linear(int numInputs, int numOutputs);

    ~Linear() override = default;
    
    /**
     * @brief Feedforward method of a linear layer class
     * @param input - The input matrix that is fed into this layer from the previous layer (previous
     *                layer's activation matrix after a pass through an activation function layer)
     * @return Output matrix (next layer's activations)
    */
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    /**
     * @brief Backpropagation method of a linear layer class
     * NOTE: The individual layer classes implementation of backward propagation are quite
     *       confusing on their own. Look at Container.cpp to see how they are all
     *       used together.
     * @param dLA - The input matrix with the derivative of loss w/ respect to NEXT layer
     *              output activations
     * @return Error w/ respect to layer's activation
    */
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dLA) override;

    /**
     * @brief Getter for name (linear)
     * @return Name string
    */
    [[nodiscard]] std::string getName() const override;

    /**
     * @brief Getter for number of input nodes
     * @return Number of input features
    */
    [[nodiscard]] int getInputs() const override;

    /**
     * @brief Getter for number of output nodes
     * @return Number of output features
    */
    [[nodiscard]] int getOutputs() const override;

    /**
     * @brief Getter for learning rate
     * @return Learning rate value
    */
    [[nodiscard]] float getLR() const override;

    /**
     * @brief Set learning rate of this individual layer
     * @param learningRate - Desired learning rate
    */
    void setLearningRate(const float& learningRate);
    
    /**
     * @brief Update weights matrix (TESTING PURPOSES)
     * @param newWeights - Matrix for weights to be updated to
    */
    void setWeight(Eigen::MatrixXf& newWeights);
    
    /**
     * @brief Update bias matrix (TESTING PURPOSES)
     * @param newBias - Matrix for biases to be updated to
    */
    void setBias(Eigen::MatrixXf& newBias);
    
    /**
     * @brief Retrieve weights matrix (TESTING PURPOSES)
     * @return Const. reference to internal weights matrix
    */
    [[nodiscard]] const Eigen::MatrixXf& getWeight() const;
    
    /**
     * @brief Retrieve bias matrix (TESTING PURPOSES)
     * @return Const. reference to internal bias matrix
    */
    [[nodiscard]] const Eigen::VectorXf& getBias() const;

private:
    Eigen::MatrixXf curActivation;
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;
    
    int inputFeatures;
    int outputFeatures;
    float eta = 0.001;
};

#endif