/**
* \file Net.hpp
*
* \brief //TODO
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include "neural/layers/Layer.hpp"
#include "neural/layers/Relu.hpp"

namespace neural {
    class Net {


    private:
        Relu<double, Eigen::Sizes<10, 1>, Eigen::Sizes<10, 1>> m_relu;
        //std::vector<Layer> m_layers;
    };
}

#endif //NEURAL_NET_HPP
