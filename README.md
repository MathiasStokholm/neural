## Neural - Modern C++11 header-only neural network library
The goal of this library is to provide a simple way of embedding neural networks in modern C++ 
code without the overhead of a full-fledged deep learning library, such as TensorFlow, Caffe, etc.

Neural makes heavy use of template metaprogramming to achieve the following:
 * Automatic differentiation for backpropagation using [Stan Math](https://github.com/stan-dev/math)
 * Compile-time checking of layer input and output sizes
 * Training-related functionality only gets included when necessary (i.e. not for inference)
 
### Example Usage
```
auto net = neural::make_net(
        neural::Linear<neural::Derivative, inputSize, numNeurons, batchSize, false>(),
        neural::Tanh<neural::Derivative, numNeurons, batchSize>(),
        neural::Linear<neural::Derivative, numNeurons, outputSize, batchSize, false>(),
        neural::Tanh<neural::Derivative, outputSize, batchSize>()
);
net.attachOptimizer(neural::OptimizerFactory::Adam(0.1));
neural::MeanSquaredError<neural::Derivative, outputSize, batchSize> error;

constexpr unsigned int numEpoch = 10;
for (unsigned int currentEpoch = 0; currentEpoch < numEpoch; currentEpoch++) {
    neural::GradientGuard guard;
    
    const auto prediction = net.forward(input);
    auto loss = error.compute(prediction, labels);
    net.backward(loss);
    
    std::cout << "Epoch: " << currentEpoch << ". Loss: " << loss << std::endl;
} 
```

### Dependencies
 * Build:
   * CMake v3.0.0+
 * Training:
   * [Stan Math Library v2.17.1](https://github.com/stan-dev/math)
     * cvodes v2.9.0 (included in Stan Math release)
     * Eigen v3.3.3 (included in Stan Math release)
     * Boost v1.64 (included in Stan Math release)
 * Inference:
   * [Eigen v3.3.3](https://github.com/eigenteam/eigen-git-mirror)
   

### Building
To use Neural in your code, simply include `#include <neural/Neural.hpp>` in your code 
and make sure all the Neural dependencies appear on your include path. Because Neural is 
header-only, there's nothing to build. 

If you use CMake, you can use Neural by adding the following to your `CMakeLists.txt`:
```
set(STAN_MATH_PATH "/path/to/stan/math" CACHE STRING "Path to the Stan Math repository")
add_subdirectory(/path/to/neural)
target_link_libraries(${YOUR_PROJECT_NAME} PRIVATE neural)
```

To run the Neural tests, invoke CMake with the appropriate option set:
```
mkdir build && cd build
cmake -D NEURAL_BUILD_TESTS=ON ..
make
```

The tests can now be run via the binary:
```
cd build
./bin/neural_tests
```  

### Examples
Neural contains the following examples:
 * mnist - Training a simple network to classify images of handwritten digits
 
To build the examples, invoke CMake with the appropriate option set:
```
mkdir build && cd build
cmake -D NEURAL_BUILD_EXAMPLES=ON ..
make
```

The examples can now be run via the binary:
```
cd build
./bin/neural_mnist
```
