/**
* \file Tensor.hpp
*
* \brief //TODO
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_TENSOR_HPP
#define NEURAL_TENSOR_HPP

#include <unsupported/Eigen/CXX11/Tensor>

namespace neural {
    template <typename Dtype, unsigned int BatchSize, unsigned int ChannelSize>
    class Tensor: public Eigen::TensorFixedSize<Dtype, Eigen::Sizes<BatchSize, ChannelSize>> {
    public:
        Tensor() = default;

        template <typename Derived>
        Tensor(const Eigen::TensorBase<Derived>& tensor):
                Eigen::TensorFixedSize<Dtype, Eigen::Sizes<BatchSize, ChannelSize>>::TensorFixedSize(tensor) {}
    };
}

#endif //NEURAL_TENSOR_HPP
