/**
* \file MeanSquaredError.hpp
*
* \brief //TODO
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_MEANSQUAREDERROR_HPP
#define NEURAL_MEANSQUAREDERROR_HPP

#include <neural/Tensor.hpp>

namespace neural {
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
