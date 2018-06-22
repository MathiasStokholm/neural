/**
* \file Relu.hpp
*
* \brief //TODO
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_RELU_HPP
#define NEURAL_RELU_HPP

#include <neural/util/Gradient.hpp>
#include <neural/Tensor.hpp>

namespace neural {
    template <typename Dtype, int InputSize, int BatchSize>
    class Relu {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;
        using OutputTensor = Tensor<Dtype, BatchSize, InputSize>;

        OutputTensor forward(const InputTensor &input) const {
            return input.cwiseMax(Dtype(0));
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type updateWeights() {
            // No weights to adjust here
        }
    };
}

#endif //NEURAL_RELU_HPP
