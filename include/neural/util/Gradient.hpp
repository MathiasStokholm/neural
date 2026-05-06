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

// autodiff reverse-mode with first-class Eigen integration (header-only)
#include <autodiff/reverse/var/eigen.hpp>

namespace neural {

/// The native autodiff reverse-mode scalar type used for automatic differentiation.
/// autodiff::var already provides Eigen NumTraits and ScalarBinaryOpTraits.
using Derivative = autodiff::var;

/// Return the scalar value stored in a Derivative node.
inline double val(const Derivative& d) { return autodiff::val(d); }

} // namespace neural

#else // AUTO_DIFF_ENABLED not set

// Inference-only mode – autodiff/gradients are not available.
// Forward declarations to keep compiler happy (these can never be called
// due to the std::enable_if guards in the layer/optimizer headers).
namespace neural {
    using Derivative = void;
}

#endif // AUTO_DIFF_ENABLED
#endif // NEURAL_GRADIENT_HPP
