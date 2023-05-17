# GroundUpNeuralNet

GroundUpNeuralNet is a fast & efficient C++ library for constructing (deep) neural networks. 

This library was built with [Eigen 3.4.0](https://eigen.tuxfamily.org/index.php?title=Main_Page) for its powerful vectorization capabilities and [GoogleTest](https://github.com/google/googletest) for unit testing. 

**NOTE:** these dependencies come w/ this project's CMake, so no manual linking needed :)

It also comes with a sample data generators to test your network (i.e. sine wave with noise) and a trainer class for easier training.

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
- GPU support w/ CUDA
