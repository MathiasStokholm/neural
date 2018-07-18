/**
* \file Net.hpp
*
* \brief Class for wrapping a set of layers into a neural network
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <neural/util/Gradient.hpp>
#include <neural/optimizers/OptimizerFactory.hpp>

namespace neural {
    namespace detail {
        /**
         * @brief Struct used to recurse calls up or down a stack of layers at compile-time
         * @tparam N The number of steps to recurse
         */
        template<size_t N>
        struct Recursor {
            template<typename Input, typename Layers>
            static inline auto update(Input && input, Layers && layers)
            -> decltype(std::get<N-1>(std::forward<Layers>(layers)).forward(Recursor<N-1>::update(std::forward<Input>(input), std::forward<Layers>(layers)))) {
                return std::get<N-1>(std::forward<Layers>(layers)).forward(Recursor<N-1>::update(std::forward<Input>(input), std::forward<Layers>(layers)));
            }

            template<typename Layers>
            static inline void backward(Layers && layers) {
                std::get<N-1>(std::forward<Layers>(layers)).updateWeights();
                Recursor<N-1>::backward(std::forward<Layers>(layers));
            }

            template<typename Layers>
            static inline void attach(OptimizerFactory && factory, Layers && layers) {
                std::get<N-1>(std::forward<Layers>(layers)).attachOptimizer(std::forward<OptimizerFactory>(factory));
                Recursor<N-1>::attach(std::forward<OptimizerFactory>(factory), std::forward<Layers>(layers));
            }
        };

        /**
         * @brief Recursor specialization for the base case
         */
        template<>
        struct Recursor<0> {
            template<typename Input, typename Layers>
            static inline Input update(Input && input, Layers && layers) {
                return input;
            }

            template<typename Layers>
            static inline void backward(Layers && layers) {
                // Noop
            }

            template<typename Layers>
            static inline void attach(OptimizerFactory && factory, Layers && layers) {
                // Noop
            }
        };

        /**
         * @brief Recursively call the forward functions of all layers, chaining inputs to outputs throughout the stack
         * @tparam Input The type of the input
         * @tparam Layers The types of the layers
         * @param input The input to the layer stack
         * @param layers The layers
         * @return The output from the final layer
         */
        template<typename Input, typename Layers>
        inline auto update(Input && input, Layers && layers)
        -> decltype(Recursor<std::tuple_size<typename std::decay<Layers>::type>::value>::update(std::forward<Input>(input), std::forward<Layers>(layers))) {
            return Recursor<std::tuple_size<typename std::decay<Layers>::type>::value>::update(std::forward<Input>(input), std::forward<Layers>(layers));
        }

        /**
         * @brief Call the backward functions of all layers
         * @tparam Layers The types of the layers
         * @param layers The layers
         */
        template<typename Layers>
        inline void backward(Layers && layers) {
            Recursor<std::tuple_size<typename std::decay<Layers>::type>::value>::backward(std::forward<Layers>(layers));
        }

        /**
         * @brief Use an OptimizerFactory to create optimizers for all layers
         * @tparam Layers The types of the layers
         * @param factory The factory to use for creating optimizers
         * @param layers The layers
         */
        template<typename Layers>
        inline void attach(OptimizerFactory && factory, Layers && layers) {
            Recursor<std::tuple_size<typename std::decay<Layers>::type>::value>::attach(std::forward<OptimizerFactory>(factory), std::forward<Layers>(layers));
        }
    }

    /**
     * @brief Class for wrapping a set of layers into a neural network
     * @tparam Layers the types of the layers to wrap
     */
    template<typename... Layers>
    class Net {
    public:
        using InputTensor = typename std::tuple_element<0, std::tuple<Layers...>>::type::InputTensor;
        using OutputTensor = typename std::tuple_element<std::tuple_size<std::tuple<Layers...>>::value-1, std::tuple<Layers...>>::type::OutputTensor;
        using Dtype = typename InputTensor::Scalar;

        /**
         * @brief Create a new Net from a set of layers
         * @param layers The layers to wrap in this Net
         */
        explicit Net(Layers&&... layers): m_layers(std::make_tuple(std::forward<Layers>(layers)...)) {}

        /**
         * @brief Propagate a given input throughout all the layers of the network and return the output
         * @param input The input to the Net
         * @return The output of the Net
         */
        OutputTensor forward(const InputTensor &input) {
            return detail::update(input, m_layers);
        }

        /**
         * @brief Attach optimizers to all layers wrapped by this Net
         * @param factory The OptimizerFactory to use for creating optimizers
         */
        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type attachOptimizer(OptimizerFactory && factory) {
            detail::attach(std::forward<OptimizerFactory>(factory), m_layers);
            m_optimizerAttached = true;
        }

        /**
         * @brief Use the provided loss to perform backpropagation through all layers to update weights
         * @param loss The loss to use for calculating gradients
         */
        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type backward(Q &loss) {
            if (!m_optimizerAttached) {
                throw std::runtime_error("No optimizer attached - cannot perform backwards pass");
            }

            // Compute and store partial derivatives with respect to the loss
            loss.grad();

            // Perform weight updates
            detail::backward(m_layers);
        }

    private:
        std::tuple<Layers...> m_layers;     ///< The stack of layers wrapped by this Net
        bool m_optimizerAttached = false;   ///< Whether an optimizer has been attached to the layers of this Net
    };

    /**
     * @brief Helper function to create a new Net with automatic template deduction
     * @tparam Layers The types of the layers to wrap in a Net
     * @param layers The layers to wrap in a Net
     * @return The created Net
     */
    template<typename... Layers>
    constexpr Net<typename std::decay<Layers>::type...> make_net(Layers&&... layers) {
        typedef Net<typename std::decay<Layers>::type...> net_type;
        return net_type(std::forward<Layers>(layers)...);
    }
}

#endif //NEURAL_NET_HPP
