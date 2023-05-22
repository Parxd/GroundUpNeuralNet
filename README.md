# GroundUpNeuralNet

GroundUpNeuralNet is a fast & efficient C++ library for constructing (deep) neural networks. 

This library was built with [Eigen 3.4.0](https://eigen.tuxfamily.org/index.php?title=Main_Page) for its powerful vectorization capabilities and [GoogleTest](https://github.com/google/googletest) for unit testing. 

**NOTE:** these dependencies come w/ this project's CMake, so no manual linking needed :)

It also comes with sample data generators to test your network (i.e. sine wave with noise), a trainer class for easier training using mini-batch optimization, and a save/load feature to save your model's weights and biases.

The following activation functions are supported:
- Sigmoid
- ReLU
- LeakyReLU
- Hyperbolic tangent
- Softmax

The following loss functions are supported:
- Mean squared error
- Categorical cross-entropy

Future features I’d like to implement:
- Support for generic numeric types, such as doubles & integers. As of now, only float32 is supported.
- Support for different optimizers
- GPU support w/ CUDA

An example run using sample data (noisy sine wave approximation):
![](https://github.com/Parxd/GroundUpNeuralNet/blob/main/res/example.png)
