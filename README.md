# GroundUpNeuralNet

GroundUpNeuralNet is a fast & efficient C++ library for constructing (deep) neural networks. 

This library was built with [Eigen 3.4.0](https://eigen.tuxfamily.org/index.php?title=Main_Page) for its powerful vectorization capabilities and [GoogleTest](https://github.com/google/googletest) for unit testing. 

**NOTE:** these dependencies come w/ this project's CMake, so no manual linking needed

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

As an example, a neural network can be constructed following this pattern:
```cpp
Container container(
              new Linear(2, 10),
              new ReLU,
              new Linear(10, 10),
              new ReLU,
              new Linear(10, 2),
              new Softmax
          );
          
// ...or using BaseModule's factory methods

Container container(
              BaseModule::make<Linear>(784, 15),
              BaseModule::make<LeakyReLU>(),
              BaseModule::make<Linear>(15, 10),
              BaseModule::make<LeakyReLU>(),
              BaseModule::make<Linear>(10, 5),
              BaseModule::make<Softmax>()
          );
```

Another example: Using the sample data generator, trainer class (w/ cross-entropy loss), and save feature:
```cpp
auto data = Sine::generate(1000000, 0.2, 200, 50, 1.1);
Eigen::MatrixXf features = data.topRows(2);
Eigen::MatrixXf labels = data.bottomRows(2);
Trainer<CE>::train(cont, features, labels, 32, 5, 1, true);
cont.save("../src/model.csv");
```

An example run using sample data (noisy sine wave approximation, batch size of 32):
![](https://github.com/Parxd/GroundUpNeuralNet/blob/main/res/example.png)

Future features I’d like to implement:
- Support for generic numeric types, such as doubles & integers. As of now, only float32 is supported.
- Support for different optimizers
- GPU support w/ CUDA
