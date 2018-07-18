/**
* \file MeanSquaredError.hpp
*
* \brief Mean Squared error loss layer
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_MEANSQUAREDERROR_HPP
#define NEURAL_MEANSQUAREDERROR_HPP

#include <neural/Tensor.hpp>

namespace neural {
    /**
     * @brief Mean Squared error loss layer. Calculates the mean squared error between a set of predictions and ground
     *        truth labels
     * @tparam Dtype The scalar type to use for this loss layer
     * @tparam InputSize The number of inputs to this loss layer
     * @tparam BatchSize The batch size to use
     */
    template <typename Dtype, unsigned int InputSize, unsigned int BatchSize>
    class MeanSquaredError {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;

        Dtype compute(const InputTensor &predictions, const InputTensor &labels) const {
            const typename InputTensor::EigenType &predictionsEigen = predictions;
            const typename InputTensor::EigenType &labelsEigen = labels;

            Eigen::Tensor<Dtype, 0> squaredSum = (predictionsEigen - labelsEigen).square().sum();
            return squaredSum(0) / Dtype(BatchSize);
        }
    };
}

#endif //NEURAL_MEANSQUAREDERROR_HPP
