/**
* \file Gradient.hpp
*
* \brief Functionality related to automatic differentiation
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_GRADIENT_HPP
#define NEURAL_GRADIENT_HPP

#ifdef AUTO_DIFF_ENABLED
#include <stan/math.hpp>

namespace neural {
    using Derivative = stan::math::var;         ///< The scalar type used for training neural networks using auto diff
    using BaseType = stan::math::var::Scalar;   ///< The scalar type underlying the neural::Derivative type

    /**
     * @brief Retrieves the gradient from a Derivative
     * @note This should only be called after evaluating the gradient using .grad()
     * @param derivative The variable to retrieve the gradient from
     * @return The retrieved gradient
     */
    inline BaseType getGradient(const Derivative &derivative) {
        return derivative.adj();
    }

    /**
     * RAII wrapper around the Stan Math memory handling functions
     * Stan Math uses an arena allocator internally to handle allocations. Unless recover_memory is called, this arena
     * will continue to grow. Because we're often only interested in keeping gradients around for a single forward+backward
     * pass, the GradientGuard is an easy helper struct that handles the necessary Stan Math calls to keep the arena at
     * a fixed size by recovering memory.
     */
    struct GradientGuard {
        /**
         * @brief Creates a new GradientGuard
         */
        GradientGuard() {
            stan::math::start_nested();
        }

        /**
         * @brief Releases a GradientGuard, causing intermediary neural::Derivative allocations to be reset
         */
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
