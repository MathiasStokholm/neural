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
    template <typename Dtype_, unsigned int BatchSize_, unsigned int ChannelSize_>
    class Tensor: public Eigen::TensorFixedSize<Dtype_, Eigen::Sizes<BatchSize_, ChannelSize_>> {
    public:
        typedef Dtype_ Dtype;
        enum {
            BatchSize = BatchSize_,
            ChannelSize = ChannelSize_
        };
        using EigenType = Eigen::TensorFixedSize<Dtype, Eigen::Sizes<BatchSize, ChannelSize>>;

        Tensor() = default;

        template <typename Derived>
        Tensor(const Eigen::TensorBase<Derived>& tensor): EigenType::TensorFixedSize(tensor) {}

        template <typename Derived>
        Tensor(Eigen::TensorBase<Derived>& tensor): EigenType::TensorFixedSize(tensor) {}

        template <typename Derived>
        Tensor(Eigen::TensorBase<Derived>&& tensor): EigenType::TensorFixedSize(tensor) {}
    };
}

#endif //NEURAL_TENSOR_HPP
