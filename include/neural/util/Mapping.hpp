/**
* \file Mapping.hpp
*
* \brief //TODO
*
* \date   Jun 18, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_MAPPING_HPP
#define NEURAL_MAPPING_HPP

#include <Eigen/Core>

namespace neural {
    template <unsigned int Rows, unsigned int Cols, class TensorIn>
    Eigen::Map<const Eigen::Matrix<typename TensorIn::CoeffReturnType, Rows, Cols>>
    ConstTensorToMatrix(const TensorIn &input) {
        return Eigen::Map<const Eigen::Matrix<typename TensorIn::CoeffReturnType, Rows, Cols>>(input.data());
    }

    template <unsigned int Size, unsigned int Stride, class TensorIn>
    Eigen::Map<const Eigen::Matrix<typename TensorIn::CoeffReturnType, Size, 1>, Eigen::Unaligned, Eigen::Stride<Size, Stride>>
    ConstTensorSliceToVector(const TensorIn &input, unsigned int slice) {
        assert(slice <= Stride);
        return Eigen::Map<const Eigen::Matrix<typename TensorIn::CoeffReturnType, Size, 1>, Eigen::Unaligned, Eigen::Stride<Size, Stride>>(input.data() + slice);
    }

    template <unsigned int Size, unsigned int Stride, class TensorIn>
    Eigen::Map<Eigen::Matrix<typename TensorIn::CoeffReturnType, Size, 1>, Eigen::Unaligned, Eigen::Stride<Size, Stride>>
    TensorSliceToVector(TensorIn &input, unsigned int slice) {
        assert(slice <= Stride);
        return Eigen::Map<Eigen::Matrix<typename TensorIn::CoeffReturnType, Size, 1>, Eigen::Unaligned, Eigen::Stride<Size, Stride>>(input.data() + slice);
    }
}

#endif //NEURAL_MAPPING_HPP
