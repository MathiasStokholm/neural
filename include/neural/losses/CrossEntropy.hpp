/**
* \file CrossEntropy.hpp
*
* \brief Cross Entropy loss layer
*
* \date   Jun 23, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_CROSSENTROPY_HPP
#define NEURAL_CROSSENTROPY_HPP

#include <neural/Tensor.hpp>
#include <neural/util/Mapping.hpp>

namespace neural {
    /**
     * @brief Cross Entropy loss layer
     * @tparam Dtype The scalar type to use for this loss layer
     * @tparam InputSize The number of inputs to this loss layer
     * @tparam BatchSize The batch size to use
     */
    template <typename Dtype, unsigned int InputSize, unsigned int BatchSize>
    class CrossEntropy {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;

        Dtype compute(const InputTensor &predictions, const InputTensor &labels) const {
            Eigen::Tensor<Dtype, 0> crossEntropy = -(labels * (predictions + Dtype(1e-9)).log()).sum();
            return crossEntropy(0) / Dtype(BatchSize);
        }

        Dtype accuracy(const InputTensor &predictions, const InputTensor &labels) const {
            const Eigen::Tensor<bool, 1> matches = (predictions.argmax(1) == labels.argmax(1));
            return ConstTensorToMatrix<BatchSize, 1>(matches).count() / Dtype(BatchSize);
        }
    };
}

#endif //NEURAL_CROSSENTROPY_HPP
