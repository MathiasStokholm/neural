/**
* \file Linear.hpp
*
* \brief //TODO
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_LINEAR_HPP
#define NEURAL_LINEAR_HPP

#include <iostream>
#include "Layer.hpp"
#include "neural/util/Gradient.hpp"
#include "neural/util/Mapping.hpp"

namespace neural {
    template <typename Dtype, unsigned int InputSize, unsigned int NumNeurons, unsigned int BatchSize>
    class Linear: Layer<Dtype, Eigen::Sizes<BatchSize, InputSize>, Eigen::Sizes<BatchSize, NumNeurons>> {
    public:
        using InputDims = Eigen::Sizes<BatchSize, InputSize>;
        using OutputDims = Eigen::Sizes<BatchSize, NumNeurons>;
        using WeightDims = Eigen::Sizes<InputSize, NumNeurons>;
        using BiasDims = Eigen::Sizes<1, NumNeurons>;
        using typename Layer<Dtype, Eigen::Sizes<BatchSize, InputSize>, Eigen::Sizes<BatchSize, NumNeurons>>::InputTensor;
        using typename Layer<Dtype, Eigen::Sizes<BatchSize, InputSize>, Eigen::Sizes<BatchSize, NumNeurons>>::OutputTensor;

        Linear() {
            m_weights.setConstant(Dtype(30));
            m_biases.setConstant(Dtype(10));
        }

        OutputTensor forward(const InputTensor &input) const override {
            // Create output
            OutputTensor result;

            // This is the standard Eigen::Tensor way of doing generalized matrix multiplication, but
            // the auto diff libraries don't like this yet!
            // static Eigen::array<Eigen::IndexPair<int>, 1> productDims = {Eigen::IndexPair<int>(1, 0)};
            // const OutputTensor result = input.contract(m_weights, productDims);

            // Instead, we apply the operations to each input in batch
            const auto mappedWeights = ConstTensorToMatrix<InputSize, NumNeurons>(m_weights).transpose();
            for (unsigned int batch = 0; batch < BatchSize; batch++) {
                // Map tensors to Eigen matrices
                const auto mappedTensor = neural::ConstTensorSliceToVector<InputSize, BatchSize>(input, batch);
                auto mappedOutput = neural::TensorSliceToVector<NumNeurons, BatchSize>(result, batch);

                // Perform y1 = Ax
                mappedOutput.noalias() = mappedWeights * mappedTensor;
            }

            if (!m_useBias) {
                return result;
            }

            // Apply bias to every element in batch by broadcasting the biases using replication
            // y2 = y1 + b
            static Eigen::array<Eigen::Index, 2> broadcastDims{BatchSize, 1};
            return result + m_biases.broadcast(broadcastDims);
        }

        template<class Q = Dtype>
        typename std::enable_if<!std::is_same<Q, Derivative>::value, void>::type updateWeights() override {
            // Regular value updateWeights() - Backprop not supported in this case
            throw std::runtime_error("Backprop only supported for Derivative type");
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type updateWeights() override {
            // Backprop is available, adjust weights and biases
            const double learningRate = 1e-4;

            // Update weights
            const auto weightGrad = m_weights.unaryExpr(std::cref(getGradient));
            m_weights -= learningRate * weightGrad;

            if (m_useBias) {
                const double biasLearningRate = 1e-4;

                // Update biases
                const auto biasGrad = m_biases.unaryExpr(std::cref(getGradient));
                m_biases -= biasLearningRate * biasGrad;
            }
        }

    private:
        bool m_useBias = true;
        Eigen::TensorFixedSize<Dtype, WeightDims> m_weights;
        Eigen::TensorFixedSize<Dtype, BiasDims> m_biases;
    };
}

#endif //NEURAL_LINEAR_HPP
