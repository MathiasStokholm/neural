/**
* \file Sigmoid.hpp
*
* \brief Sigmoid activation function 1 / (1 + e^(-x))
*
* \date   Jun 23, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_SIGMOID_HPP
#define NEURAL_SIGMOID_HPP

#include <neural/util/Gradient.hpp>
#include <neural/Tensor.hpp>
#include <neural/optimizers/OptimizerFactory.hpp>

namespace neural {
    /**
     * @brief Sigmoid activation function 1 / (1 + e^(-x))
     * @tparam Dtype The scalar type to use for this layer
     * @tparam InputSize The number of inputs to this layer
     * @tparam BatchSize The batch size to use
     */
    template <typename Dtype, int InputSize, int BatchSize>
    class Sigmoid {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;
        using OutputTensor = Tensor<Dtype, BatchSize, InputSize>;

        OutputTensor forward(const InputTensor &input) const {
            return (Dtype(0.5) * (Dtype(0.5) * input).tanh() + Dtype(0.5)).eval();
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

#endif //NEURAL_SIGMOID_HPP
