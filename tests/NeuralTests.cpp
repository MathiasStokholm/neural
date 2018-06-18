#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "neural/layers/Relu.hpp"
#include "neural/layers/Linear.hpp"
#include "neural/util/Gradient.hpp"

#include <Eigen/Eigen>


TEST_CASE("Testing ReLu", "[relu]" ) {
    constexpr int inputSize = 10;
    constexpr int batchSize = 1;

    // Create input and output tensors
    Eigen::TensorFixedSize<neural::Derivative, Eigen::Sizes<batchSize, inputSize>> x, expectedDerivativesX;
    x.setValues({{-10, -7, -5, -3, 0, 1, 3,  5,  7, 10}});

    neural::Relu<neural::Derivative, inputSize, batchSize> relu;

    expectedDerivativesX.setValues({{0, 0, 0, 0, 1, 1, 1, 1, 1, 1}});

    // Perform operations
    neural::Relu<neural::Derivative, inputSize, batchSize>::OutputTensor d = relu.forward(x);

    // Evaluate gradient
    Eigen::Tensor<neural::Derivative, 0> y = d.sum();
    y(0).grad();

    // Check derivatives
    for (unsigned int i = 0; i < inputSize; i++) {
        const auto grad = x(i).adj();
        REQUIRE( expectedDerivativesX(i) == grad );
    }
}

TEST_CASE("Testing backprop", "[backprop]" ) {
    constexpr int inputSize = 10;
    constexpr int numNeurons = 5;
    constexpr int numNeurons2 = 30;
    constexpr int batchSize = 1;

    // Create input and output tensors
    Eigen::TensorFixedSize<neural::Derivative, Eigen::Sizes<batchSize, inputSize>> input;
    input.setValues({{30, -3, -2, -1, 0, 1, 3,  5,  7, 10}});

    neural::Linear<neural::Derivative, inputSize, numNeurons, batchSize> linear;
    neural::Relu<neural::Derivative, numNeurons2, batchSize> relu;
    neural::Linear<neural::Derivative, numNeurons, numNeurons2, batchSize> linear2;

    // Perform operations
    Eigen::Tensor<neural::Derivative, 0> y;
    for (int i=0; i<10; i++) {
        const auto result1 = linear.forward(input);
        const auto result2 = linear2.forward(result1);
        const auto output = relu.forward(result2);
        std::cout << "ReLu results: " << output << std::endl;

        y = output.sum();
        y(0).grad();

        linear.updateWeights();
        stan::math::set_zero_all_adjoints();
    }
}
