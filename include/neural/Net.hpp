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

        template<typename Input, typename Layers>
        inline auto update(Input && input, Layers && layers)
        -> decltype(Recursor<std::tuple_size<typename std::decay<Layers>::type>::value>::update(std::forward<Input>(input), std::forward<Layers>(layers))) {
            return Recursor<std::tuple_size<typename std::decay<Layers>::type>::value>::update(std::forward<Input>(input), std::forward<Layers>(layers));
        }

        template<typename Layers>
        inline void backward(Layers && layers) {
            Recursor<std::tuple_size<typename std::decay<Layers>::type>::value>::backward(std::forward<Layers>(layers));
        }

        template<typename Layers>
        inline void attach(OptimizerFactory && factory, Layers && layers) {
            Recursor<std::tuple_size<typename std::decay<Layers>::type>::value>::attach(std::forward<OptimizerFactory>(factory), std::forward<Layers>(layers));
        }
    }

    template<typename... Layers>
    class Net {
    public:
        using InputTensor = typename std::tuple_element<0, std::tuple<Layers...>>::type::InputTensor;
        using OutputTensor = typename std::tuple_element<std::tuple_size<std::tuple<Layers...>>::value-1, std::tuple<Layers...>>::type::OutputTensor;
        using Dtype = typename InputTensor::Scalar;

        explicit Net(Layers&&... layers): m_layers(std::make_tuple(std::forward<Layers>(layers)...)) {}

        OutputTensor forward(const InputTensor &input) {
            return detail::update(input, m_layers);
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type attachOptimizer(OptimizerFactory && factory) {
            detail::attach(std::forward<OptimizerFactory>(factory), m_layers);
            m_optimizerAttached = true;
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type backward(Q &loss) {
            if (!m_optimizerAttached) {
                throw std::runtime_error("No optimizer attached - cannot perform backwards pass");
            }

            loss.grad();

            // Perform weight updates
            detail::backward(m_layers);
        }

    private:
        std::tuple<Layers...> m_layers;
        bool m_optimizerAttached = false;
    };

    template<typename... Layers>
    constexpr Net<typename std::decay<Layers>::type...> make_net(Layers&&... layers) {
        typedef Net<typename std::decay<Layers>::type...> net_type;
        return net_type(std::forward<Layers>(layers)...);
    }
}

#endif //NEURAL_NET_HPP
