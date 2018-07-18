/**
* \file SGD.hpp
*
* \brief Optimizer implementing regular gradient descent
*
* \date   Jun 22, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_SGD_HPP
#define NEURAL_SGD_HPP

#include <neural/optimizers/Optimizer.hpp>
#include <neural/util/Gradient.hpp>

namespace neural {
    /**
     * @brief Optimizer implementing regular gradient descent with momentum
     * @tparam Tensor The type of tensor to optimize
     */
    template <typename Tensor>
    class SGDOptimizer: public Optimizer<Tensor> {
    public:
        using GradTensor = typename Optimizer<Tensor>::GradTensor;

        SGDOptimizer(double learningRate, double momentum):
                m_learningRate(learningRate), m_momentum(momentum), m_lastUpdate() {
            m_lastUpdate.setZero();
        }

        GradTensor update(const Tensor &tensor) override {
            const auto grad = tensor.unaryExpr(std::cref(getGradient));
            m_lastUpdate = m_momentum * m_lastUpdate + m_learningRate * grad;
            return m_lastUpdate;
        }

    private:
        double m_learningRate;
        double m_momentum;
        GradTensor m_lastUpdate;
    };
}

#endif //NEURAL_SGD_HPP
