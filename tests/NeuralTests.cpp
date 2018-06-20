#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <Eigen/Core>
#include "neural/Tensor.hpp"
#include "neural/layers/Relu.hpp"
#include "neural/layers/Linear.hpp"
#include "neural/util/Gradient.hpp"
#include "neural/util/Mapping.hpp"
#include "neural/Net.hpp"

#include <Eigen/Eigen>

TEST_CASE("Testing Net", "[net]" ) {
    constexpr int inputSize = 10;
    constexpr int batchSize = 1;

    // Create input and output tensors
    neural::Tensor<neural::Derivative, batchSize, inputSize> x, expectedDerivativesX;
    x.setValues({{-10, -7, -5, -3, 0, 1, 3,  5,  7, 10}});
    expectedDerivativesX.setValues({{0, 0, 0, 0, 1, 1, 1, 1, 1, 1}});

    auto net = neural::make_net(
            neural::Relu<neural::Derivative, inputSize, batchSize>(),
            neural::Relu<neural::Derivative, inputSize, batchSize>()
    );
    const auto result = net.forward(x);

    // Evaluate gradient
    net.backward(result);

    for (unsigned int i = 0; i < inputSize; i++) {
        const auto grad = x(i).adj();
        REQUIRE( expectedDerivativesX(i) == grad );
    }
}

TEST_CASE("Testing ReLu", "[relu]" ) {
    constexpr int inputSize = 10;
    constexpr int batchSize = 1;

    // Create input and output tensors
    neural::Tensor<neural::Derivative, batchSize, inputSize> x, expectedDerivativesX;
    x.setValues({{-10, -7, -5, -3, 0, 1, 3,  5,  7, 10}});
    expectedDerivativesX.setValues({{0, 0, 0, 0, 1, 1, 1, 1, 1, 1}});

    // Perform operations
    neural::Relu<neural::Derivative, inputSize, batchSize> relu;
    const auto d = relu.forward(x);

    // Evaluate gradient
    Eigen::Tensor<neural::Derivative, 0> y = d.sum();
    y(0).grad();

    // Check derivatives
    for (unsigned int i = 0; i < inputSize; i++) {
        const auto grad = x(i).adj();
        REQUIRE( expectedDerivativesX(i) == grad );
    }
}

TEST_CASE("Testing tensor -> matrix/vector mapping functions", "[mapping]" ) {
    constexpr int inputSize = 3;
    constexpr int batchSize = 2;

    // Create tensor
    neural::Tensor<double, batchSize, inputSize> tensor;
    tensor.setValues({{10, 10, 10}, {-30, -30, -30}});

    const auto map1 = neural::TensorSliceToVector<inputSize, batchSize>(tensor, 0);
    const auto map2 = neural::TensorSliceToVector<inputSize, batchSize>(tensor, 1);
    const auto map1Const = neural::ConstTensorSliceToVector<inputSize, batchSize>(tensor, 0);
    const auto map2Const = neural::ConstTensorSliceToVector<inputSize, batchSize>(tensor, 1);
    const auto map3 = neural::ConstTensorToMatrix<batchSize, inputSize>(tensor);

    for (int i = 0; i < inputSize; i++) {
        REQUIRE( tensor(0, i) == map1(i) );
        REQUIRE( tensor(0, i) == map1Const(i) );
        REQUIRE( tensor(1, i) == map2(i) );
        REQUIRE( tensor(1, i) == map2Const(i) );

        for (int j = 0; j < batchSize; j++) {
            REQUIRE( tensor(j, i) == map3(j, i) );
        }
    }
}

TEST_CASE("Testing backprop", "[backprop]" ) {
    constexpr int inputSize = 10;
    constexpr int numNeurons = 5;
    constexpr int numNeurons2 = 30;
    constexpr int batchSize = 2;

    // Create input and output tensors
    neural::Tensor<neural::Derivative, batchSize, inputSize> input;
    input.setValues({{30, -3, -2, -1, 0, 1, 3,  5,  7, 10}, {30, -3, -2, -1, 0, 1, 3,  5,  7, 10}});

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
        linear2.updateWeights();
        stan::math::set_zero_all_adjoints();
    }
}
