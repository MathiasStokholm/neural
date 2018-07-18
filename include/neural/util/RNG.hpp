/**
* \file RNG.hpp
*
* \brief Simple random number generator
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_RNG_HPP
#define NEURAL_RNG_HPP

#include <random>

namespace neural {
    /**
     * @brief Simple random number generator
     */
    class RNG {
    public:
        /**
         * @brief Create a new RNG
         * @param min The minimum value to generate
         * @param max The maximum value to generate
         */
        RNG(int min, int max): m_generator(std::random_device()()), m_distribution(min, max) {}

        /**
         * @brief Draw a new sample from the random generator
         * @return A new sample from the random generator
         */
        int getNext() {
            return m_distribution(m_generator);
        }

    private:
        std::mt19937 m_generator;
        std::uniform_int_distribution<int> m_distribution;
    };
}

#endif //NEURAL_RNG_HPP
