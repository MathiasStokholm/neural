/**
* \file Linear.hpp
*
* \brief Layer for applying a linear operation, e.g. y = Ax + b, where x is the input, A is a set of learned
*        weights, and b is a set of learned biases
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_LINEAR_HPP
#define NEURAL_LINEAR_HPP

#include <neural/util/Gradient.hpp>
#include <neural/util/Mapping.hpp>
#include <neural/Tensor.hpp>
#include <neural/initializers/GlorotNormal.hpp>
#include <neural/optimizers/OptimizerFactory.hpp>

namespace neural {
    /**
     * @brief Layer for applying a linear operation, e.g. y = Ax + b, where x is the input, A is a set of learned
     *        weights, and b is a set of learned biases
     * @tparam Dtype The scalar type to use for this layer
     * @tparam InputSize The number of inputs to this layer
     * @tparam NumNeurons The number of neurons (outputs)
     * @tparam BatchSize The batch size to use
     * @tparam UseBias Whether to include a bias term in this linear layer
     */
    template <typename Dtype, unsigned int InputSize, unsigned int NumNeurons, unsigned int BatchSize, bool UseBias=true>
    class Linear {
    public:
        using InputTensor = Tensor<Dtype, BatchSize, InputSize>;
        using OutputTensor = Tensor<Dtype, BatchSize, NumNeurons>;
        using WeightsTensor = Tensor<Dtype, InputSize, NumNeurons>;
        using BiasesTensor = Tensor<Dtype, 1, NumNeurons>;
        enum {
            HasBias = UseBias
        };

        Linear(): m_optimizerAttached(false) {
            // Initialize weights with a GlorotNormal initialization
            // TODO: Support other initialization types through a template parameter
            m_weights.template setRandom<GlorotNormal<Dtype, InputSize, NumNeurons>>();

            if (HasBias) {
                m_biases.setConstant(0);
            }
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type attachOptimizer(const OptimizerFactory &factory) {
            m_weightsOptimizer = factory.createOptimizer(m_weights);
            if (HasBias) {
                m_biasOptimizer = factory.createOptimizer(m_biases);
            }
            m_optimizerAttached = true;
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
            for (unsigned int i = 0; i < BatchSize; i++) {
                // Map tensors to Eigen matrices
                const auto mappedTensor = ConstTensorSliceToVector<InputSize, BatchSize>(input, i);
                auto mappedOutput = TensorSliceToVector<NumNeurons, BatchSize>(result, i);

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
            if (!m_optimizerAttached) {
                throw std::runtime_error("No optimizer attached - cannot update weights");
            }

            // Backprop is available, adjust weights and biases
            m_weights -= m_weightsOptimizer->update(m_weights);
            if (HasBias) {
                m_biases -= m_biasOptimizer->update(m_biases);
            }
        }

    private:
        WeightsTensor m_weights;    ///< The weights of this linear layer
        std::unique_ptr<Optimizer<WeightsTensor>> m_weightsOptimizer;   ///< Pointer to an optimizer used for updating the weights
        BiasesTensor m_biases;      ///< The biases of this linear layer
        std::unique_ptr<Optimizer<BiasesTensor>> m_biasOptimizer;       ///< Pointer to an optimizer used for updating the biases
        bool m_optimizerAttached;   ///< Whether an optimizer has been attached to this layer
    };
}

#endif //NEURAL_LINEAR_HPP
