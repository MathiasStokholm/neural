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

#include <neural/util/Gradient.hpp>
#include <neural/util/Mapping.hpp>
#include <neural/Tensor.hpp>
#include <neural/initializers/NormalInitializer.hpp>

namespace neural {
    template <typename Dtype, unsigned int InputSize, unsigned int NumNeurons, unsigned int BatchSize, bool UseBias>
    class Linear {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;
        using OutputTensor = Tensor<Dtype, BatchSize, NumNeurons>;
        using WeightsTensor = Tensor<Dtype, InputSize, NumNeurons>;
        using BiasesTensor = Tensor<Dtype, 1, NumNeurons>;
        enum {
            HasBias = UseBias
        };

        Linear() {
            m_weights.template setRandom<NormalInitializer<Dtype>>();

            if (HasBias) {
                m_biases.template setRandom<NormalInitializer<Dtype>>();
            }
        }

        OutputTensor forward(const InputTensor &input) const {
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
                const auto mappedTensor = ConstTensorSliceToVector<InputSize, BatchSize>(input, batch);
                auto mappedOutput = TensorSliceToVector<NumNeurons, BatchSize>(result, batch);

                // Perform y1 = Ax
                mappedOutput.noalias() = mappedWeights * mappedTensor;
            }

            if (!HasBias) {
                return result;
            }

            // Apply bias to every element in batch by broadcasting the biases using replication
            // y2 = y1 + b
            static Eigen::array<Eigen::Index, 2> broadcastDims{BatchSize, 1};
            return result + m_biases.broadcast(broadcastDims);
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type updateWeights() {
            // Backprop is available, adjust weights and biases
            const double learningRate = 0.1;

            // Update weights
            const auto weightGrad = m_weights.unaryExpr(std::cref(getGradient));
            m_weights -= learningRate * weightGrad;

            if (HasBias) {
                const double biasLearningRate = 0.1;

                // Update biases
                const auto biasGrad = m_biases.unaryExpr(std::cref(getGradient));
                m_biases -= biasLearningRate * biasGrad;
            }
        }

    private:
        WeightsTensor m_weights;
        BiasesTensor m_biases;
    };
}

#endif //NEURAL_LINEAR_HPP
