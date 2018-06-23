/**
* \file CrossEntropy.hpp
*
* \brief //TODO
*
* \date   Jun 23, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_CROSSENTROPY_HPP
#define NEURAL_CROSSENTROPY_HPP

#include <neural/Tensor.hpp>

namespace neural {
    template <typename Dtype, unsigned int InputSize, unsigned int BatchSize>
    class CrossEntropy {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;

        Dtype compute(const InputTensor &predictions, const InputTensor &labels) const {
            Eigen::Tensor<Dtype, 0> crossEntropy = -(labels * (predictions + Dtype(1e-9)).log()).sum();
            return crossEntropy(0) / Dtype(BatchSize);
        }
    };
}

#endif //NEURAL_CROSSENTROPY_HPP
