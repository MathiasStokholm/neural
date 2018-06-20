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

#include <stan/math.hpp>

namespace neural {
    using Derivative = stan::math::var;
    using BaseType = stan::math::var::Scalar;

    BaseType getGradient(const Derivative &derivative) {
        return derivative.adj();
    }

    void setGradientsZero() {
        stan::math::set_zero_all_adjoints();
    }
}

#endif //NEURAL_GRADIENT_HPP
