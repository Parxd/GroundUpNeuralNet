#ifndef LAYER_H
#define LAYER_H

#include "BaseModule.h"
#include "Visitor.h"

class Layer : public BaseModule
{
public:
    Layer(int numInputs, int numOutputs);
    ~Layer() = default;
    /**
     * @brief Feedforward method of a linear layer class
     * @param [input] The input matrix that is fed into this layer from the previous layer 
     * @param [output] The output matrix after operations w/ weights & biases
     * @return [NONE] Modifies output matrix in-place
    */
    void forward(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) override;
    /**
     * @brief Backpropagation method of a linear layer class
     * @param [dEW] Derivative of error with respect to weight
     * @param [output] The output matrix after operation with updated weights
     * @return [NONE] Modifies output matrix (& weights/biases) in-place
    */
    void backward(const Eigen::MatrixXf& dEW, Eigen::MatrixXf& output) override;

    /**
     * @brief Set learning rate of this individual layer
     * @param [learningRate] Desired learning rate
    */
    void setLearningRate(const float& learningRate);

    /**
     * @brief Update weights matrix (primarily for testing)
     * @param [newWeights] Matrix for weights to be updated to
    */
    void setWeight(const Eigen::MatrixXf& newWeights);

    /**
     * @brief Update bias matrix (primarily for testing)
     * @param [newBias] Matrix for biases to be updated to
    */
    void setBias(const Eigen::MatrixXf& newBias);

    /**
     * @brief Retrieve weights matrix
     * @return Const. reference to internal weights matrix
    */
    const Eigen::MatrixXf& getWeight() const;

    /**
     * @brief Retrieve bias matrix
     * @return Const. reference to internal bias matrix.
    */
    const Eigen::MatrixXf& getBias() const;

    /**
     * @brief Accept the Visitor class to access attributes
    */
    void accept(Visitor& visit) const;

private:
    Eigen::VectorXf storedInput;
    Eigen::MatrixXf weights;
    Eigen::MatrixXf bias;
    
    int inputFeatures;
    int outputFeatures;
    float eta;
};

#endif