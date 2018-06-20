/**
* \file RNG.hpp
*
* \brief //TODO
*
* \date   Jun 20, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_RNG_HPP
#define NEURAL_RNG_HPP

#include <random>

namespace neural {
    class RNG {
    public:
        RNG(int min, int max): m_generator(std::random_device()()), m_distribution(min, max) {}

        int getNext() {
            return m_distribution(m_generator);
        }

    private:
        std::mt19937 m_generator;
        std::uniform_int_distribution<int> m_distribution;
    };
}

#endif //NEURAL_RNG_HPP
