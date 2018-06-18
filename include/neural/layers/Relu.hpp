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

#include "Layer.hpp"
#include "neural/util/Gradient.hpp"

namespace neural {
    template <typename Dtype, unsigned int InputSize, unsigned int BatchSize>
    class Relu: Layer<Dtype, Eigen::Sizes<BatchSize, InputSize>, Eigen::Sizes<BatchSize, InputSize>> {
    public:
        using typename Layer<Dtype, Eigen::Sizes<BatchSize, InputSize>, Eigen::Sizes<BatchSize, InputSize>>::InputTensor;
        using typename Layer<Dtype, Eigen::Sizes<BatchSize, InputSize>, Eigen::Sizes<BatchSize, InputSize>>::OutputTensor;

        OutputTensor forward(const InputTensor &input) const override {
            return input.cwiseMax(Dtype(0));
        }
    };
}

#endif //NEURAL_RELU_HPP
