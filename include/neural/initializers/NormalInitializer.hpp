/**
* \file NormalInitializer.hpp
*
* \brief //TODO
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_NORMALINITIALIZER_HPP
#define NEURAL_NORMALINITIALIZER_HPP

#include <Eigen/Core>
#include <random>

namespace neural {
    template <typename Dtype>
    class NormalInitializer {
    public:
        NormalInitializer(): m_distribution(0, 1e-3) {}
        NormalInitializer(const NormalInitializer&) = default;

        Dtype operator()(Eigen::DenseIndex element_location,
                         Eigen::DenseIndex /*unused*/ = 0) const {
            // FIXME: This is massively hacky and ugly
            auto* nonConstThis = const_cast<NormalInitializer*>(this);
            return static_cast<Dtype>(nonConstThis->m_distribution(nonConstThis->m_generator));
        }

    private:
        std::mt19937 m_generator;
        std::normal_distribution<double> m_distribution;
    };
}

#endif //NEURAL_NORMALINITIALIZER_HPP
