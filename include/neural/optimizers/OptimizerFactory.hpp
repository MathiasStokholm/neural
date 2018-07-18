/**
* \file OptimizerFactory.hpp
*
* \brief Helper class used to instantiate optimizers for one or more layers in a net
*
* \date   Jun 22, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_OPTIMIZERFACTORY_HPP
#define NEURAL_OPTIMIZERFACTORY_HPP

#include <neural/optimizers/Optimizer.hpp>
#include <neural/optimizers/SGD.hpp>
#include <neural/optimizers/Adam.hpp>

namespace neural {
    /**
     * @brief Helper class used to instantiate optimizers for one or more layers in a net
     */
    class OptimizerFactory {
    public:
        /**
         * @brief The type of optimizer to instantiate
         */
        enum class Type {
            SGD,    ///< Regular gradient descent
            Adam    ///< Adam optimizer
        };

        /**
         * @brief Create a new OptimizerFactory that will instantiate SGD optimizers for all layers
         * @param learningRate The learning rate to use
         * @param momentum The momentum to use
         * @return a new OptimizerFactory
         */
        static inline OptimizerFactory SGD(double learningRate, double momentum=0.9) {
            return OptimizerFactory(learningRate, momentum);
        }

        /**
         * @brief Create a new OptimizerFactory that will instantiate Adam optimizers for all layers
         * @param learningRate The learning rate to use
         * @param beta1 The beta1 value to use (see Adam.hpp)
         * @param beta2 The beta2 value to use (see Adam.hpp)
         * @param epsilon The epsilon value to use (see Adam.hpp)
         * @return a new OptimizerFactory
         */
        static inline OptimizerFactory Adam(double learningRate, double beta1=0.9, double beta2=0.999, double epsilon=1e-8) {
            return OptimizerFactory(learningRate, beta1, beta2, epsilon);
        }

        /**
         * @brief Create a new optimizer for a given tensor
         * @tparam Tensor The type of the tensor to create an optimizer for
         * @param tensor The tensor to create an optimizer for
         * @return The created optimizer
         */
        template <typename Tensor>
        std::unique_ptr<Optimizer<Tensor>> createOptimizer(const Tensor &tensor) const {
            switch (m_type) {
                case Type::SGD:
                    return std::unique_ptr<Optimizer<Tensor>>(new SGDOptimizer<Tensor>(m_learningRate, m_momentum));
                case Type::Adam:
                    return std::unique_ptr<Optimizer<Tensor>>(new AdamOptimizer<Tensor>(m_learningRate, m_beta1, m_beta2, m_epsilon));
            }

            // This cannot happen, but we use this to silence compiler warnings
            throw std::runtime_error("No optimizer returned");
        }

    protected:
        /**
         * @brief Create a new SGD optimizer with the given learning rate and momentum
         * @param learningRate The learning rate to use
         * @param momentum The momentum to use
         */
        OptimizerFactory(double learningRate, double momentum):
                m_type(Type::SGD), m_learningRate(learningRate), m_momentum(momentum) {}

        /**
         * @brief Create a new Adam optimizer
         * @param learningRate The learning rate to use
         * @param beta1 The beta1 value to use (see Adam.hpp)
         * @param beta2 The beta2 value to use (see Adam.hpp)
         * @param epsilon The epsilon value to use (see Adam.hpp)
         */
        OptimizerFactory(double learningRate, double beta1, double beta2, double epsilon):
                m_type(Type::Adam), m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon) {}


        Type m_type;
        double m_learningRate;

        // SGD specific quantities
        double m_momentum = 0;

        // Adam specific quantities
        double m_beta1 = 0;
        double m_beta2 = 0;
        double m_epsilon = 0;
    };
}

#endif //NEURAL_OPTIMIZERFACTORY_HPP
