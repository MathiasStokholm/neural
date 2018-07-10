/**
* \file NormalInitializer.hpp
*
* \brief Glorot normal initializer, as described in http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_NORMALINITIALIZER_HPP
#define NEURAL_NORMALINITIALIZER_HPP

#include <Eigen/Core>
#include <random>

namespace neural {
    /**
     * @brief Glorot normal initializer (also called Xavier normal initializer)
     * Draws samples from a truncated normal distribution centered on 0 with the std deviation set according to the
     * number of inputs and outputs.
     * @tparam Dtype: The data type to generate samples as
     * @tparam FanIn: The number of input units of the tensor to generate weights for
     * @tparam FanOut: The number of output units of the tensor to generate weights for
     */
    template <typename Dtype, unsigned int FanIn, unsigned int FanOut>
    class GlorotNormal {
    public:
        GlorotNormal(): m_distribution(0.0, std::sqrt(2.0 / (FanIn + FanOut))) {}
        GlorotNormal(const GlorotNormal&) = default;

        Dtype operator()(Eigen::DenseIndex element_location, Eigen::DenseIndex /*unused*/ = 0) const {
            // FIXME: This is massively hacky and ugly
            auto* nonConstThis = const_cast<GlorotNormal*>(this);
            return Dtype(nonConstThis->m_distribution(nonConstThis->m_generator));
        }

    private:
        std::mt19937 m_generator;
        std::normal_distribution<double> m_distribution;
    };
}

#endif //NEURAL_NORMALINITIALIZER_HPP
