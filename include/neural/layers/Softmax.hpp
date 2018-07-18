/**
* \file Softmax.hpp
*
* \brief Softmax activation function
*
* \date   Jun 23, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_SOFTMAX_HPP
#define NEURAL_SOFTMAX_HPP

#include <neural/util/Gradient.hpp>
#include <neural/Tensor.hpp>
#include <neural/optimizers/OptimizerFactory.hpp>

namespace neural {
    /**
     * @brief Softmax activation function
     * @tparam Dtype The scalar type to use for this layer
     * @tparam InputSize The number of inputs to this layer
     * @tparam BatchSize The batch size to use
     */
    template <typename Dtype, int InputSize, int BatchSize>
    class Softmax {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;
        using OutputTensor = Tensor<Dtype, BatchSize, InputSize>;

        OutputTensor forward(const InputTensor &input) const {
            // Find max to subtract from input - this makes the solution more numerically stable
            const auto shiftedInput = input - input.maximum(Eigen::array<int, 1>{1}).eval()
                    .reshape(Eigen::array<int, 2>{BatchSize, 1})
                    .broadcast(Eigen::array<int, 2>{1, InputSize});
            const auto exponentiatedInput = shiftedInput.exp();
            const auto output = exponentiatedInput / exponentiatedInput.sum(Eigen::array<int, 1>{1}).eval()
                    .reshape(Eigen::array<int, 2>({BatchSize, 1}))
                    .broadcast(Eigen::array<int, 2>({1, InputSize}));
            return output;
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type attachOptimizer(const OptimizerFactory &factory) {
            // No weights to optimize
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type updateWeights() {
            // No weights to adjust here
        }
    };
}

#endif //NEURAL_SOFTMAX_HPP
