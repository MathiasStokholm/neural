/**
* \file Gradient.hpp
*
* \brief //TODO
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_GRADIENT_HPP
#define NEURAL_GRADIENT_HPP

#ifdef AUTO_DIFF_ENABLED
#include <stan/math.hpp>

namespace neural {
    using Derivative = stan::math::var;
    using BaseType = stan::math::var::Scalar;

    BaseType getGradient(const Derivative &derivative) {
        return derivative.adj();
    }

    struct GradientGuard {
        GradientGuard() {
            stan::math::start_nested();
        }

        ~GradientGuard() {
            stan::math::set_zero_all_adjoints_nested();
            stan::math::recover_memory_nested();
        }
    };
}

#else

// Inference-only mode - autodiff/gradients are not available
// Forward declarations to keep compiler happy (these can never be called due to std::enable_if usage)
namespace neural {
    using Derivative = void;
    void getGradient();
}

#endif //AUTO_DIFF_ENABLED
#endif //NEURAL_GRADIENT_HPP
