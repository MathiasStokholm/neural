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

namespace neural {
    class OptimizerFactory {
    public:
        enum class Type {
            SGD
        };

        explicit OptimizerFactory(double learningRate, double momentum=0.9):
                m_type(Type::SGD), m_learningRate(learningRate), m_momentum(momentum) {}

        template <typename Tensor>
        std::unique_ptr<Optimizer<Tensor>> createOptimizer(const Tensor &tensor) const {
            switch (m_type) {
                case Type::SGD:
                    return std::unique_ptr<Optimizer<Tensor>>(new SGD<Tensor>(m_learningRate, m_momentum));
            }

            // This cannot happen, but we use this to silence compiler warnings
            throw std::runtime_error("No optimizer returned");
        }

    private:
        Type m_type;
        double m_learningRate;
        double m_momentum;
    };
}

#endif //NEURAL_OPTIMIZERFACTORY_HPP
