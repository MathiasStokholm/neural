/**
* \file Neural.hpp
*
* \brief Top level include to grab everything in neural
*
* \date   Jun 28, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_NEURAL_HPP
#define NEURAL_NEURAL_HPP

#include <neural/Tensor.hpp>
#include <neural/Net.hpp>

#include <neural/layers/Linear.hpp>
#include <neural/layers/Relu.hpp>
#include <neural/layers/Sigmoid.hpp>
#include <neural/layers/Softmax.hpp>
#include <neural/layers/Tanh.hpp>
#include <neural/losses/CrossEntropy.hpp>
#include <neural/losses/MeanSquaredError.hpp>

#include <neural/optimizers/SGD.hpp>
#include <neural/optimizers/Adam.hpp>

#include <neural/util/Gradient.hpp>
#include <neural/util/Mapping.hpp>
#include <neural/util/RNG.hpp>

#endif //NEURAL_NEURAL_HPP
