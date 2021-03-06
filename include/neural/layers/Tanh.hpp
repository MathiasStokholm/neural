/**
* \file Tanh.hpp
*
* \brief Hyperbolic tangent activation function
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_TANH_HPP
#define NEURAL_TANH_HPP

#include <neural/util/Gradient.hpp>
#include <neural/Tensor.hpp>
#include <neural/optimizers/OptimizerFactory.hpp>

namespace neural {
    /**
     * @brief Hyperbolic tangent activation function
     * @tparam Dtype The scalar type to use for this layer
     * @tparam InputSize The number of inputs to this layer
     * @tparam BatchSize The batch size to use
     */
    template <typename Dtype, int InputSize, int BatchSize>
    class Tanh {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;
        using OutputTensor = Tensor<Dtype, BatchSize, InputSize>;

        OutputTensor forward(const InputTensor &input) const {
            return input.tanh().eval();
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

#endif //NEURAL_TANH_HPP
