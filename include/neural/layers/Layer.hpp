/**
* \file Layer.hpp
*
* \brief //TODO
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_LAYER_HPP
#define NEURAL_LAYER_HPP

#include <unsupported/Eigen/CXX11/Tensor>

namespace neural {
    template <typename Dtype, typename InputDims, typename OutputDims>
    class Layer {
    public:
        using InputTensor = Eigen::TensorFixedSize<Dtype, InputDims>;
        using OutputTensor = Eigen::TensorFixedSize<Dtype, OutputDims>;

        virtual OutputTensor forward(const InputTensor &input) const = 0;

        virtual void updateWeights() {};
    };
}

#endif //NEURAL_LAYER_HPP
