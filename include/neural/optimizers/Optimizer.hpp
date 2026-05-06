/**
* \file Optimizer.hpp
*
* \brief Base class for all optimizers
*
* \date   Jun 22, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_OPTIMIZER_HPP
#define NEURAL_OPTIMIZER_HPP

#include <neural/Tensor.hpp>

namespace neural {
    /**
     * @brief Base class for all optimizers
     * @tparam Tensor The type of tensor to optimize
     */
    template <typename Tensor>
    class Optimizer {
    public:
        using GradTensor = neural::Tensor<double, Tensor::BatchSize, Tensor::ChannelSize>;

        virtual GradTensor update(const GradTensor &grad) = 0;
    };
}

#endif //NEURAL_OPTIMIZER_HPP
