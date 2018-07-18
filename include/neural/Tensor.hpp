/**
* \file Tensor.hpp
*
* \brief The tensor type used by Neural to enable compile-time layer size checking
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_TENSOR_HPP
#define NEURAL_TENSOR_HPP

#include <unsupported/Eigen/CXX11/Tensor>

namespace neural {
    /**
     * @brief The tensor type used by Neural to enable compile-time layer size checking
     * @tparam Dtype_ The scalar type stored by this tensor
     * @tparam BatchSize_ The batch size of this tensor
     * @tparam ChannelSize_ The channel size of this tensor
     */
    template <typename Dtype_, unsigned int BatchSize_, unsigned int ChannelSize_>
    class Tensor: public Eigen::TensorFixedSize<Dtype_, Eigen::Sizes<BatchSize_, ChannelSize_>> {
    public:
        typedef Dtype_ Dtype;           ///< The scalar type stored by this tensor
        enum {
            BatchSize = BatchSize_,     ///< The batch size of this tensor
            ChannelSize = ChannelSize_  ///< The channel size of this tensor
        };
        using EigenType = Eigen::TensorFixedSize<Dtype, Eigen::Sizes<BatchSize, ChannelSize>>;  ///< The underlying Eigen Tensor type

        Tensor() = default;

        /**
         * @brief Implicit conversion function allowing a corresponding Eigen Tensor to be converted into a neural::Tensor
         */
        template <typename Derived>
        Tensor(const Eigen::TensorBase<Derived> &tensor): EigenType::TensorFixedSize(tensor) {}
    };
}

#endif //NEURAL_TENSOR_HPP
