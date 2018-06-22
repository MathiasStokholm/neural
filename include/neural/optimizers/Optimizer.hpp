/**
* \file Optimizer.hpp
*
* \brief //TODO
*
* \date   Jun 22, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_OPTIMIZER_HPP
#define NEURAL_OPTIMIZER_HPP

namespace neural {
    template <typename Tensor>
    class Optimizer {
    public:
        using GradTensor = neural::Tensor<double, Tensor::BatchSize, Tensor::ChannelSize>;

        virtual GradTensor update(const Tensor &tensor) = 0;
    };
}

#endif //NEURAL_OPTIMIZER_HPP