/**
* \file main.cpp
*
* \brief Example showing training of a simple neural network on the MNIST dataset (http://yann.lecun.com/exdb/mnist).
*        This training example reaches ~92% accuracy after just a few epochs of training.
*
* \date   Jun 28, 2018
* \author Mathias Bøgh Stokholm
*/

#include <iostream>
#include <mnist/mnist_reader.hpp>
#include <neural/Neural.hpp>

constexpr unsigned int batchSize = 100;
constexpr unsigned int inputSize = 28*28;
constexpr unsigned int outputSize = 10;
constexpr unsigned int epochs = 15;

// Data types used for training
using InputTensor = neural::Tensor<neural::Derivative, batchSize, inputSize>;
using OutputTensor = neural::Tensor<neural::Derivative, batchSize, outputSize>;

/**
 * @brief Normalize the values in a dataset to be in the [0, 1] range
 * @param data [in/out]: The dataset to normalize
 */
void normalize(std::vector<std::vector<double>> &data) {
    // Find max pixel value over all images (likely 255)
    double maxValue = 0;
    for (const auto &image: data) {
        const auto maxImagePixel = *std::max_element(image.begin(), image.end());
        if (maxImagePixel > maxValue) {
            maxValue = maxImagePixel;
        }
    }

    // Normalize all pixels to [0-1] range
    for (auto &image: data) {
        for (auto &pixel: image) {
            pixel /= maxValue;
        }
    }
}

int main(int argc, char* argv[]) {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data and normalize data
    auto dataset = mnist::read_dataset<std::vector, std::vector, double, std::uint8_t>(MNIST_DATA_LOCATION);
    normalize(dataset.training_images);
    normalize(dataset.test_images);

    // Create easy getter function for data
    auto loadData = [&dataset] (bool training, unsigned int startIndex) {
        InputTensor x;
        OutputTensor y;
        for (unsigned int j = 0; j < batchSize; j++) {
            // Map image at index into input tensor and convert to neural::Derivative
            const auto index = startIndex + j;
            const auto &images = training? dataset.training_images: dataset.test_images;
            auto xMapped = neural::TensorSliceToVector<inputSize, batchSize>(x, j);
            const auto inputMapped = Eigen::Map<const Eigen::Matrix<double, inputSize, 1>>(images[index].data());
            xMapped = inputMapped.cast<neural::Derivative>();

            // Map label at index into output tensor using one-hot encoding
            const auto &labels = training? dataset.training_labels: dataset.test_labels;
            auto yMapped = neural::TensorSliceToVector<outputSize, batchSize>(y, j);
            yMapped = Eigen::Matrix<neural::Derivative, outputSize, 1>::Zero();
            yMapped[labels.at(index)] = 1;
        }
        return std::make_tuple(std::move(x), std::move(y));
    };

    // Create RNG engine and shuffling indexes
    auto rng = std::default_random_engine {};
    std::vector<unsigned int> indexes(dataset.training_images.size());
    std::iota(indexes.begin(), indexes.end(), 0U);

    // Create network and attach optimizer
    auto net = neural::make_net(
            neural::Linear<neural::Derivative, InputTensor::ChannelSize, OutputTensor::ChannelSize, batchSize>(),
            neural::Softmax<neural::Derivative, OutputTensor::ChannelSize, batchSize>()
    );
    net.attachOptimizer(neural::OptimizerFactory::Adam(0.1));

    // Create loss function
    neural::CrossEntropy<neural::Derivative, OutputTensor::ChannelSize, batchSize> error;


    //// Training section below
    const auto trainSteps = dataset.training_images.size() / batchSize;
    const auto testSteps = dataset.test_images.size() / batchSize;
    for (unsigned int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch: " << epoch << ". Performing test..." << std::endl;

        // Step over all test data
        std::vector<double> accuracies = {};
        for (unsigned int i = 0; i < testSteps; i++) {
            neural::GradientGuard guard;

            // Get input/output tensors
            InputTensor x;
            OutputTensor y;
            std::tie(x, y) = loadData(false, i * batchSize);

            // Perform forward
            const auto prediction = net.forward(x);

            // Determine error
            accuracies.emplace_back(error.accuracy(prediction, y).val());
        }
        const auto accuracy = std::accumulate(accuracies.begin(), accuracies.end(), 0.0) / accuracies.size();
        std::cout << "Mean test accuracy: " << accuracy << std::endl;

        // Shuffle indexes
        std::shuffle(indexes.begin(), indexes.end(), rng);

        // Step over all training data
        std::vector<double> losses = {};
        for (unsigned int i = 0; i < trainSteps; i++) {
            neural::GradientGuard guard;

            // Get input/output tensors
            InputTensor x;
            OutputTensor y;
            std::tie(x, y) = loadData(true, i * batchSize);

            // Perform forward
            const auto prediction = net.forward(x);

            // Determine error
            auto loss = error.compute(prediction, y);
            losses.emplace_back(loss.val());

            // Update weights
            net.backward(loss);
        }
        const auto meanLoss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
        std::cout << "Mean train loss: " << meanLoss << std::endl;
    }
}
