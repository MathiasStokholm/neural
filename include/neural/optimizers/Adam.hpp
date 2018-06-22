/**
* \file Adam.hpp
*
* \brief //TODO
*
* \date   Jun 22, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_ADAM_HPP
#define NEURAL_ADAM_HPP

#include <neural/optimizers/Optimizer.hpp>
#include <neural/util/Gradient.hpp>

namespace neural {
    template <typename Tensor>
    class AdamOptimizer: public Optimizer<Tensor> {
    public:
        using GradTensor = typename Optimizer<Tensor>::GradTensor;

        AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon):
                m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_currentStep(1) {
            m_firstMoment.setZero();
            m_secondMoment.setZero();
        }

        GradTensor update(const Tensor &tensor) override {
            const auto grad = tensor.unaryExpr(std::cref(getGradient));

            // Calculate first and second moments (mean and uncentered variance)
            m_firstMoment = m_beta1 * m_firstMoment + (1 - m_beta1) * grad;
            m_secondMoment = m_beta2 * m_secondMoment + (1 - m_beta2) * grad.square();

            // Apply bias corrections
            const auto firstMomentCorrected = m_firstMoment / (1 - std::pow(m_beta1, m_currentStep));
            const auto secondMomentCorrected = m_secondMoment / (1 - std::pow(m_beta2, m_currentStep));

            // Update current step
            m_currentStep++;

            // Return update
            return (m_learningRate / (secondMomentCorrected.sqrt() + m_epsilon)) * firstMomentCorrected;
        }

    private:
        double m_learningRate;
        double m_beta1;
        double m_beta2;
        double m_epsilon;
        GradTensor m_firstMoment{};
        GradTensor m_secondMoment{};
        unsigned int m_currentStep;
    };
}

#endif //NEURAL_ADAM_HPP
