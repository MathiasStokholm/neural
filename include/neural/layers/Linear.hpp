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
                // Initialize biases to zero; each element gets its own expression node.
                for (unsigned int i = 0; i < NumNeurons; i++) {
                    m_biases(0, i) = Dtype(0);
                }
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
            // Map input and weights to Eigen matrices (zero-copy) and compute the
            // full batch matrix multiplication in one call.
            const auto inputMat    = ConstTensorToMatrix<BatchSize, InputSize>(input);
            const auto weightsMat  = ConstTensorToMatrix<InputSize, NumNeurons>(m_weights);

            OutputTensor result;
            Eigen::Map<Eigen::Matrix<Dtype, BatchSize, NumNeurons>>(result.data()) =
                inputMat * weightsMat;

            if (!HasBias) {
                return result;
            }

            // Apply bias to every element in batch by broadcasting the biases using replication
            // y2 = y1 + b
            static Eigen::array<Eigen::Index, 2> broadcastDims{BatchSize, 1};
            return result + m_biases.broadcast(broadcastDims);
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type updateWeights(const Q& loss) {
            if (!m_optimizerAttached) {
                throw std::runtime_error("No optimizer attached - cannot update weights");
            }

            // Map the flat weight storage to an Eigen vector so autodiff::gradient()
            // can compute ∂loss/∂w for all weights in a single pass.
            Eigen::Map<Eigen::Matrix<Q, InputSize * NumNeurons, 1>> wMap(m_weights.data());
            const auto wGradVec = autodiff::gradient(loss, wMap);

            // Pack the gradient vector into a GradTensor for the optimizer.
            using WGradTensor = Tensor<double, InputSize, NumNeurons>;
            WGradTensor wGrad;
            Eigen::Map<Eigen::Matrix<double, InputSize * NumNeurons, 1>>(wGrad.data()) = wGradVec;

            const auto wUpdate = m_weightsOptimizer->update(wGrad);

            // Extract current values, subtract update, reset to fresh independent leaves.
            using WVec = Eigen::Matrix<double, InputSize * NumNeurons, 1>;
            const WVec newWVals =
                wMap.template cast<double>() -
                Eigen::Map<const WVec>(wUpdate.data());
            wMap = newWVals.template cast<Q>();

            if (HasBias) {
                Eigen::Map<Eigen::Matrix<Q, NumNeurons, 1>> bMap(m_biases.data());
                const auto bGradVec = autodiff::gradient(loss, bMap);

                using BGradTensor = Tensor<double, 1, NumNeurons>;
                BGradTensor bGrad;
                Eigen::Map<Eigen::Matrix<double, NumNeurons, 1>>(bGrad.data()) = bGradVec;

                const auto bUpdate = m_biasOptimizer->update(bGrad);

                using BVec = Eigen::Matrix<double, NumNeurons, 1>;
                const BVec newBVals =
                    bMap.template cast<double>() -
                    Eigen::Map<const BVec>(bUpdate.data());
                bMap = newBVals.template cast<Q>();
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
