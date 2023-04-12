#ifndef LINEAR_H
#define LINEAR_H

#include "BaseModule.h"

// Forward declaration
class BaseModule;

class Linear : public BaseModule
{
public:
    /**
     * @brief Sole constructor of an individual fully-connected layer class
     * @param numInputs - The number of incoming neuron activations from the previous layer 
     * @param numOutputs - The number of outgoing neuron activations--also the number of nodes of this layer
    */
    Linear(int numInputs, int numOutputs, float lR);

    ~Linear() = default;
    
    /**
     * @brief Feedforward method of a linear layer class
     * @param input - The input matrix that is fed into this layer from the previous layer 
     * @param output - The output matrix after operations w/ weights & biases
     * @return [NONE] Modifies output matrix in-place
    */
    void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;
    
    /**
     * @brief Backpropagation method of a linear layer class
     * @param dEW - Derivative of error with respect to weight
     * @param output - The output matrix after operation with updated weights
     * @return [NONE] Modifies output matrix (& weights/biases) in-place
    */
    void backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output) override;

    /**
     * @brief Returns a string description of this linear layer
     * @return Description string
    */
    std::string description() const override;
    
    /**
     * @brief Set learning rate of this individual layer
     * @param learningRate - Desired learning rate
    */
    void setLearningRate(const float& learningRate);
    
    /**
     * @brief Update weights matrix (primarily for testing)
     * @param newWeights - Matrix for weights to be updated to
    */
    void setWeight(const Eigen::MatrixXf& newWeights);
    
    /**
     * @brief Update bias matrix (primarily for testing)
     * @param newBias - Matrix for biases to be updated to
    */
    void setBias(const Eigen::MatrixXf& newBias);
    
    /**
     * @brief Retrieve weights matrix
     * @return Const. reference to internal weights matrix
    */
    [[nodiscard]] const Eigen::MatrixXf& getWeight() const;
    
    /**
     * @brief Retrieve bias matrix
     * @return Const. reference to internal bias matrix.
    */
    [[nodiscard]] const Eigen::MatrixXf& getBias() const;

private:
    Eigen::VectorXf storedInput;
    Eigen::MatrixXf weights;
    Eigen::MatrixXf bias;
    
    int inputFeatures;
    int outputFeatures;
    float eta;
};

#endif