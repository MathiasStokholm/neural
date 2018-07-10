#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <neural/Neural.hpp>

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

TEST_CASE("Testing activation functions", "[activations]" ) {
    constexpr int numInputs = 7;
    neural::Tensor<double, 1, numInputs> x, expectedValues;
    x.setValues({{-10, -5, -1, 0, 1, 5, 10}});

    // Test ReLu
    expectedValues.setValues({{0, 0, 0, 0, 1, 5, 10}});
    neural::Relu<double, numInputs, 1> relu;
    auto result = relu.forward(x);
    for (unsigned int i = 0; i < numInputs; i++) {
        REQUIRE( expectedValues(i) == result(i) );
    }

    // Test Sigmoid
    expectedValues.setValues({{4.53978687e-05, 6.69285092e-03, 2.68941421e-01, 5.00000000e-01, 7.31058579e-01,
                               9.93307149e-01, 9.99954602e-01}});
    neural::Sigmoid<double, numInputs, 1> sigmoid;
    result = sigmoid.forward(x);
    for (unsigned int i = 0; i < numInputs; i++) {
        REQUIRE( expectedValues(i) == Approx(result(i)) );
    }

    // Test Tanh
    expectedValues.setValues({{-1, -0.9999092, -0.76159416, 0, 0.76159416, 0.9999092, 1}});
    neural::Tanh<double, numInputs, 1> tanh;
    result = tanh.forward(x);
    for (unsigned int i = 0; i < numInputs; i++) {
        REQUIRE( expectedValues(i) == Approx(result(i)) );
    }

    // Test Softmax
    expectedValues.setValues({{2.04698081e-09, 3.03798888e-07, 1.65868573e-05, 4.50877527e-05, 1.22561219e-04,
                               6.69161581e-03, 9.93123843e-01}});
    neural::Softmax<double, numInputs, 1> softmax;
    result = softmax.forward(x);
    for (unsigned int i = 0; i < numInputs; i++) {
        REQUIRE( expectedValues(i) == Approx(result(i)) );
    }
}

TEST_CASE("Testing loss functions", "[losses]" ) {
    constexpr int numInputs = 4;
    neural::Tensor<double, 1, numInputs> predictions, labels;

    // Test Cross Entropy
    predictions.setValues({{0.25, 0.0, 0.25, 0.5}});
    labels.setValues({{0, 0, 1, 0}});
    double expectedValue = 1.3862943571198905;
    neural::CrossEntropy<double, numInputs, 1> crossEntropy;
    auto result = crossEntropy.compute(predictions, labels);
    REQUIRE( expectedValue == result );
    auto accuracy = crossEntropy.accuracy(predictions, labels);
    REQUIRE( accuracy == 0 );
}

TEST_CASE("Testing net forward", "[net_forward]" ) {
    constexpr int inputSize = 10;
    constexpr int batchSize = 1;

    // Create input and output tensors
    neural::Tensor<double, batchSize, inputSize> x, expectedValues;
    x.setValues({{-10, -7, -5, -3, 0, 1, 3,  5,  7, 10}});
    expectedValues.setValues({{0, 0, 0, 0, 0, 1, 3,  5,  7, 10}});

    auto net = neural::make_net(
            neural::Relu<double, inputSize, batchSize>(),
            neural::Relu<double, inputSize, batchSize>(),
            neural::Relu<double, inputSize, batchSize>()
    );
    const auto result = net.forward(x);

    for (unsigned int i = 0; i < inputSize; i++) {
        REQUIRE( expectedValues(i) == result(i) );
    }
}

#ifdef AUTO_DIFF_ENABLED
TEST_CASE("Testing net backward", "[net_backward]" ) {
    neural::GradientGuard guard;
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
    net.attachOptimizer(neural::OptimizerFactory::SGD(0.1));

    const auto result = net.forward(x);

    // Evaluate gradient (loss is just sum)
    Eigen::Tensor<neural::Derivative, 0> loss = result.sum();
    net.backward(loss(0));

    for (unsigned int i = 0; i < inputSize; i++) {
        const auto grad = x(i).adj();
        REQUIRE( expectedDerivativesX(i) == grad );
    }
}

TEST_CASE("Testing ReLu", "[relu]" ) {
    neural::GradientGuard guard;
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

    // Attach optimizers
    neural::OptimizerFactory optimizerFactory = neural::OptimizerFactory::SGD(0.1);
    linear.attachOptimizer(optimizerFactory);
    linear2.attachOptimizer(optimizerFactory);

    // Perform operations
    Eigen::Tensor<neural::Derivative, 0> y;
    for (int i=0; i<10; i++) {
        neural::GradientGuard guard;
        const auto result1 = linear.forward(input);
        const auto result2 = linear2.forward(result1);
        const auto output = relu.forward(result2);
        std::cout << "ReLu results: " << output << std::endl;

        y = output.sum();
        y(0).grad();

        linear.updateWeights();
        linear2.updateWeights();
    }
}

TEST_CASE("Testing XOR", "[xor]" ) {
    constexpr int inputSize = 2;
    constexpr int batchSize = 1;
    constexpr int outputSize = 1;

    // XOR dataset
    neural::Tensor<neural::Derivative, 4, 2> xs;
    xs.setValues({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    neural::Tensor<neural::Derivative, 4, 1> ys;
    ys.setValues({{0}, {1}, {1}, {0}});
    neural::RNG rng(0, 3);

    // Data types used for training
    using InputTensor = neural::Tensor<neural::Derivative, batchSize, inputSize>;
    using OutputTensor = neural::Tensor<neural::Derivative, batchSize, outputSize>;

    // Create network
    auto net = neural::make_net(
            neural::Linear<neural::Derivative, InputTensor::ChannelSize, 8, batchSize, false>(),
            neural::Tanh<neural::Derivative, 8, batchSize>(),
            neural::Linear<neural::Derivative, 8, 1, batchSize, false>(),
            neural::Tanh<neural::Derivative, OutputTensor::ChannelSize, batchSize>()
    );
    net.attachOptimizer(neural::OptimizerFactory::SGD(0.1, 0.0));

    // Create loss function
    neural::MeanSquaredError<neural::Derivative, OutputTensor::ChannelSize, batchSize> error;

    // Train
    for (int i = 0; i < 500; i++) {
        neural::GradientGuard guard;

        // Get input/output tensors
        int index = rng.getNext();
        Eigen::array<int, 2> offsets = {index, 0};
        Eigen::array<int, 2> extents = {1, inputSize};
        InputTensor x = xs.slice(offsets, extents).eval();

        offsets = {index, 0};
        extents = {1, outputSize};
        OutputTensor y = ys.slice(offsets, extents).eval();

        // Perform forward
        const auto prediction = net.forward(x);

        // Determine error
        auto loss = error.compute(prediction, y);

        // Update weights
        net.backward(loss);

        // Print results
        std::cout << "Input: " << x << ", prediction: " << prediction << ", truth: " << y << std::endl;
    }

    // Test network
    for (int i = 0; i < 4; i++) {
        neural::GradientGuard guard;
        Eigen::array<int, 2> offsets = {i, 0};
        Eigen::array<int, 2> extents = {1, inputSize};
        InputTensor x = xs.slice(offsets, extents).eval();

        offsets = {i, 0};
        extents = {1, outputSize};
        OutputTensor y = ys.slice(offsets, extents).eval();

        const auto prediction = net.forward(x);
        REQUIRE( static_cast<int>(std::round(prediction(0).val())) == static_cast<int>(y(0).val()) );
    }
}
#endif //AUTO_DIFF_ENABLED
