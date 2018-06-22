/**
* \file OptimizerFactory.hpp
*
* \brief //TODO
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
    class OptimizerFactory {
    public:
        enum class Type {
            SGD,
            Adam
        };

        static inline OptimizerFactory SGD(double learningRate, double momentum=0.9) {
            return OptimizerFactory(learningRate, momentum);
        }

        static inline OptimizerFactory Adam(double learningRate, double beta1=0.9, double beta2=0.999, double epsilon=1e-8) {
            return OptimizerFactory(learningRate, beta1, beta2, epsilon);
        }

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
        OptimizerFactory(double learningRate, double momentum):
                m_type(Type::SGD), m_learningRate(learningRate), m_momentum(momentum) {}

        OptimizerFactory(double learningRate, double beta1, double beta2, double epsilon):
                m_type(Type::Adam), m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon) {}


        Type m_type;
        double m_learningRate;
        double m_momentum = 0;

        // Adam specific quantities
        double m_beta1 = 0;
        double m_beta2 = 0;
        double m_epsilon = 0;
    };
}

#endif //NEURAL_OPTIMIZERFACTORY_HPP
