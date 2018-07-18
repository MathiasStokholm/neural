/**
* \file Mapping.hpp
*
* \brief Functions for mapping from tensors to Eigen matrices
*
* \date   Jun 18, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_MAPPING_HPP
#define NEURAL_MAPPING_HPP

#include <Eigen/Core>

namespace neural {
    /**
     * @brief Maps a const tensor to an Eigen Matrix
     * @tparam Rows The number of rows in the input tensor
     * @tparam Cols The number of cols in the input tensor
     * @tparam TensorIn The type of the input tensor
     * @param input The input tensor
     * @return The tensor mapped to an Eigen Matrix
     */
    template <unsigned int Rows, unsigned int Cols, class TensorIn>
    Eigen::Map<const Eigen::Matrix<typename TensorIn::CoeffReturnType, Rows, Cols>>
    ConstTensorToMatrix(const TensorIn &input) {
        return Eigen::Map<const Eigen::Matrix<typename TensorIn::CoeffReturnType, Rows, Cols>>(input.data());
    }

    /**
     * @brief Maps a slice of a const tensor to a const Eigen Matrix (vector)
     * @tparam Size The number of elements to map (size of output vector)
     * @tparam Stride The size of the stride of the input tensor
     * @tparam TensorIn The type of the input tensor
     * @param input The input tensor
     * @param slice The index at which to slice the tensor
     * @return A slice of the tensor mapped to a const Eigen Matrix
     */
    template <unsigned int Size, unsigned int Stride, class TensorIn>
    Eigen::Map<const Eigen::Matrix<typename TensorIn::CoeffReturnType, Size, 1>, Eigen::Unaligned, Eigen::Stride<Size, Stride>>
    ConstTensorSliceToVector(const TensorIn &input, unsigned int slice) {
        assert(slice <= Stride);
        return Eigen::Map<const Eigen::Matrix<typename TensorIn::CoeffReturnType, Size, 1>, Eigen::Unaligned, Eigen::Stride<Size, Stride>>(input.data() + slice);
    }

    /**
     * @brief Maps a slice of a tensor to an Eigen Matrix (vector)
     * @tparam Size The number of elements to map (size of output vector)
     * @tparam Stride The size of the stride of the input tensor
     * @tparam TensorIn The type of the input tensor
     * @param input The input tensor
     * @param slice The index at which to slice the tensor
     * @return A slice of the tensor mapped to an Eigen Matrix
     */
    template <unsigned int Size, unsigned int Stride, class TensorIn>
    Eigen::Map<Eigen::Matrix<typename TensorIn::CoeffReturnType, Size, 1>, Eigen::Unaligned, Eigen::Stride<Size, Stride>>
    TensorSliceToVector(TensorIn &input, unsigned int slice) {
        assert(slice <= Stride);
        return Eigen::Map<Eigen::Matrix<typename TensorIn::CoeffReturnType, Size, 1>, Eigen::Unaligned, Eigen::Stride<Size, Stride>>(input.data() + slice);
    }
}

#endif //NEURAL_MAPPING_HPP
