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

namespace neural {
    namespace detail {
        //// Templates related to forward calls
        template<size_t N>
        struct Call {
            template<typename Input, typename Layers>
            static inline auto call(Input && input, Layers && layers)
            -> decltype(std::get<N-1>(std::forward<Layers>(layers)).forward(Call<N-1>::call(std::forward<Input>(input), std::forward<Layers>(layers)))) {
                return std::get<N-1>(std::forward<Layers>(layers)).forward(Call<N-1>::call(std::forward<Input>(input), std::forward<Layers>(layers)));
            }
        };

        template<>
        struct Call<0> {
            template<typename Input, typename Layers>
            static inline auto call(Input && input, Layers && layers) -> decltype(input) {
                return input;
            }
        };

        template<typename Input, typename Layers>
        inline auto call(Input && input, Layers && layers)
        -> decltype(Call<std::tuple_size<typename std::decay<Layers>::type>::value>::call(std::forward<Input>(input), std::forward<Layers>(layers))) {
            return Call<std::tuple_size<typename std::decay<Layers>::type>::value>::call(std::forward<Input>(input), std::forward<Layers>(layers));
        }
        ////

        //// Templates related to backward calls
        template<size_t N>
        struct Backward {
            template<typename Layers>
            static inline void backward(Layers && layers) {
                std::get<N-1>(std::forward<Layers>(layers)).updateWeights();
                Backward<N-1>::backward(std::forward<Layers>(layers));
            }
        };

        template<>
        struct Backward<0> {
            template<typename Layers>
            static inline void backward(Layers && layers) {
                // Noop
            }
        };

        template<typename Layers>
        inline auto backward(Layers && layers)
        -> decltype(Backward<std::tuple_size<typename std::decay<Layers>::type>::value>::backward(std::forward<Layers>(layers))) {
            return Backward<std::tuple_size<typename std::decay<Layers>::type>::value>::backward(std::forward<Layers>(layers));
        }
        ////
    }

    template<typename... Layers>
    class Net {
    public:
        using InputTensor = typename std::tuple_element<0, std::tuple<Layers...>>::type::InputTensor;
        using OutputTensor = typename std::tuple_element<std::tuple_size<std::tuple<Layers...>>::value-1, std::tuple<Layers...>>::type::OutputTensor;
        using Dtype = typename InputTensor::Scalar;

        explicit Net(Layers&&... layers): m_layers(std::make_tuple(std::forward<Layers>(layers)...)) {}

        OutputTensor forward(const InputTensor &input) {
            return detail::call(input, m_layers);
        }

        template<class Q = Dtype>
        typename std::enable_if<std::is_same<Q, Derivative>::value, void>::type backward(Q &loss, bool zeroGradients=true) {
            loss.grad();

            // Perform weight updates
            detail::backward(m_layers);

            // Zero all gradients
            if (zeroGradients) {
                setGradientsZero();
            }
        }

    private:
        std::tuple<Layers...> m_layers;
    };

    template<typename... Layers>
    constexpr Net<typename std::decay<Layers>::type...> make_net(Layers&&... layers) {
        typedef Net<typename std::decay<Layers>::type...> net_type;
        return net_type(std::forward<Layers>(layers)...);
    }
}

#endif //NEURAL_NET_HPP
