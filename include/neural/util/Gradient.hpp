/**
* \file Gradient.hpp
*
* \brief Functionality related to automatic differentiation
*
* \date   Jun 13, 2018
* \author Mathias Bøgh Stokholm
*/

#ifndef NEURAL_GRADIENT_HPP
#define NEURAL_GRADIENT_HPP

#ifdef AUTO_DIFF_ENABLED

#include <memory>
#include <vector>
#include <cmath>
#include <limits>
#include <ostream>

// autodiff reverse-mode expression tree (header-only)
#include <autodiff/reverse/var/var.hpp>

namespace neural {

namespace detail {
using ExprPtr = autodiff::reverse::detail::ExprPtr<double>;
using IndepExpr = autodiff::reverse::detail::IndependentVariableExpr<double>;

// Thread-local registry of weak_ptrs to every leaf gradient storage.
// resetAllAdjs() zeros all live entries and prunes expired ones.
inline std::vector<std::weak_ptr<double>>& adjRegistry() {
    static thread_local std::vector<std::weak_ptr<double>> registry;
    return registry;
}

inline void registerAdj(const std::shared_ptr<double>& adj) {
    adjRegistry().push_back(adj);
}

inline void resetAllAdjs() {
    auto& reg = adjRegistry();
    std::size_t dst = 0;
    for (std::size_t i = 0; i < reg.size(); ++i) {
        if (auto sp = reg[i].lock()) {
            *sp = 0.0;
            reg[dst++] = std::move(reg[i]);
        }
        // expired entries are simply dropped
    }
    reg.resize(dst);
}
} // namespace detail

/**
 * @brief Differentiable scalar type used when training neural networks.
 *
 * Each leaf (created from a numeric value) owns an expression node of type
 * IndependentVariableExpr.  The node's gradPtr is bound to m_adj so that
 * during backpropagation it directly accumulates the adjoint.  Copies share
 * the same expression node and the same gradient storage, so reading adj()
 * on the original or any copy always gives the same value.
 *
 * Intermediate results (produced by arithmetic) carry a freshly-allocated,
 * unbound m_adj; nobody writes to it and adj() on an intermediate is
 * meaningless (but harmless).
 */
class Derivative {
public:
    detail::ExprPtr m_expr;          ///< autodiff expression node
    std::shared_ptr<double> m_adj;   ///< gradient storage (shared with copies)

    // ------------------------------------------------------------------ //
    //  Constructors
    // ------------------------------------------------------------------ //

    /// Default: leaf with value 0
    Derivative() : Derivative(0.0) {}

    /// Leaf constructed from any arithmetic type
    template<typename U,
             typename = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    Derivative(U val)
        : m_expr(std::make_shared<detail::IndepExpr>(static_cast<double>(val)))
        , m_adj(std::make_shared<double>(0.0))
    {
        m_expr->bind_value(m_adj.get());
        detail::registerAdj(m_adj);
    }

    /// Copy: share expression node and gradient storage
    Derivative(const Derivative& other) = default;

    // ------------------------------------------------------------------ //
    //  Assignment
    // ------------------------------------------------------------------ //

    /// Copy-assign: unbind old gradPtr first, then share the source's node
    Derivative& operator=(const Derivative& other) {
        if (this != &other) {
            m_expr->bind_value(nullptr);   // prevent stale gradPtr after reassign
            m_expr = other.m_expr;
            m_adj  = other.m_adj;
        }
        return *this;
    }

    /// Assign from arithmetic: replace with a fresh leaf
    template<typename U,
             typename = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    Derivative& operator=(U val) {
        m_expr->bind_value(nullptr);
        m_expr = std::make_shared<detail::IndepExpr>(static_cast<double>(val));
        m_adj  = std::make_shared<double>(0.0);
        m_expr->bind_value(m_adj.get());
        detail::registerAdj(m_adj);
        return *this;
    }

    // ------------------------------------------------------------------ //
    //  Value and gradient access
    // ------------------------------------------------------------------ //

    /// Current value of this node in the expression tree
    double val() const { return m_expr->val; }

    /// Accumulated adjoint (meaningful only on leaf nodes after grad())
    double adj() const { return *m_adj; }

    /// Trigger reverse-mode backpropagation from this node.
    /// Resets all registered leaf adjoints to zero first.
    void grad() const {
        detail::resetAllAdjs();
        m_expr->propagate(1.0);
    }

    /// Explicit conversion to double
    explicit operator double() const { return val(); }

    // ------------------------------------------------------------------ //
    //  In-place arithmetic operators
    // ------------------------------------------------------------------ //

    Derivative& operator+=(const Derivative& rhs) {
        *this = *this + rhs;
        return *this;
    }
    Derivative& operator-=(const Derivative& rhs) {
        *this = *this - rhs;
        return *this;
    }
    Derivative& operator*=(const Derivative& rhs) {
        *this = *this * rhs;
        return *this;
    }
    Derivative& operator/=(const Derivative& rhs) {
        *this = *this / rhs;
        return *this;
    }

    /// In-place subtract by a plain double (used for weight updates).
    /// Produces a fresh independent leaf so the expression chain stays short.
    Derivative& operator-=(double d) {
        *this = Derivative(val() - d);
        return *this;
    }

    // ------------------------------------------------------------------ //
    //  Binary arithmetic operators
    // ------------------------------------------------------------------ //

    friend Derivative operator+(const Derivative& a, const Derivative& b) {
        return Derivative(a.m_expr + b.m_expr);
    }
    friend Derivative operator-(const Derivative& a, const Derivative& b) {
        return Derivative(a.m_expr - b.m_expr);
    }
    friend Derivative operator*(const Derivative& a, const Derivative& b) {
        return Derivative(a.m_expr * b.m_expr);
    }
    friend Derivative operator/(const Derivative& a, const Derivative& b) {
        return Derivative(a.m_expr / b.m_expr);
    }

    friend Derivative operator+(const Derivative& a, double b) { return Derivative(a.m_expr + b); }
    friend Derivative operator-(const Derivative& a, double b) { return Derivative(a.val() - b); }
    friend Derivative operator*(const Derivative& a, double b) { return Derivative(a.m_expr * b); }
    friend Derivative operator/(const Derivative& a, double b) { return Derivative(a.m_expr / b); }

    friend Derivative operator+(double a, const Derivative& b) { return Derivative(a + b.m_expr); }
    friend Derivative operator-(double a, const Derivative& b) { return Derivative(a - b.m_expr); }
    friend Derivative operator*(double a, const Derivative& b) { return Derivative(a * b.m_expr); }
    friend Derivative operator/(double a, const Derivative& b) { return Derivative(a / b.m_expr); }

    friend Derivative operator-(const Derivative& a) { return Derivative(-a.m_expr); }

    // ------------------------------------------------------------------ //
    //  Comparison operators (return bool for Eigen max/min ops)
    // ------------------------------------------------------------------ //

    friend bool operator==(const Derivative& a, const Derivative& b) { return a.val() == b.val(); }
    friend bool operator!=(const Derivative& a, const Derivative& b) { return a.val() != b.val(); }
    friend bool operator< (const Derivative& a, const Derivative& b) { return a.val() <  b.val(); }
    friend bool operator<=(const Derivative& a, const Derivative& b) { return a.val() <= b.val(); }
    friend bool operator> (const Derivative& a, const Derivative& b) { return a.val() >  b.val(); }
    friend bool operator>=(const Derivative& a, const Derivative& b) { return a.val() >= b.val(); }

    friend bool operator==(const Derivative& a, double b) { return a.val() == b; }
    friend bool operator!=(const Derivative& a, double b) { return a.val() != b; }
    friend bool operator< (const Derivative& a, double b) { return a.val() <  b; }
    friend bool operator<=(const Derivative& a, double b) { return a.val() <= b; }
    friend bool operator> (const Derivative& a, double b) { return a.val() >  b; }
    friend bool operator>=(const Derivative& a, double b) { return a.val() >= b; }

    friend bool operator==(double a, const Derivative& b) { return a == b.val(); }
    friend bool operator!=(double a, const Derivative& b) { return a != b.val(); }
    friend bool operator< (double a, const Derivative& b) { return a <  b.val(); }
    friend bool operator<=(double a, const Derivative& b) { return a <= b.val(); }
    friend bool operator> (double a, const Derivative& b) { return a >  b.val(); }
    friend bool operator>=(double a, const Derivative& b) { return a >= b.val(); }

    // ------------------------------------------------------------------ //
    //  Stream output
    // ------------------------------------------------------------------ //

    friend std::ostream& operator<<(std::ostream& os, const Derivative& x) {
        return os << x.val();
    }

    /// Construct a dependent (intermediate) node directly from an expression.
    /// Used internally by arithmetic operators and math functions.
    explicit Derivative(detail::ExprPtr e)
        : m_expr(std::move(e))
        , m_adj(std::make_shared<double>(0.0))
    {}
};

// ---------------------------------------------------------------------- //
//  Math functions – found by ADL when Derivative is used with Eigen ops  //
// ---------------------------------------------------------------------- //

inline Derivative exp  (const Derivative& x) { return Derivative(autodiff::reverse::detail::exp  (x.m_expr)); }
inline Derivative log  (const Derivative& x) { return Derivative(autodiff::reverse::detail::log  (x.m_expr)); }
inline Derivative tanh (const Derivative& x) { return Derivative(autodiff::reverse::detail::tanh (x.m_expr)); }
inline Derivative sqrt (const Derivative& x) { return Derivative(autodiff::reverse::detail::sqrt (x.m_expr)); }
inline Derivative abs  (const Derivative& x) { return Derivative(autodiff::reverse::detail::abs  (x.m_expr)); }
inline Derivative pow  (const Derivative& x, double e) {
    return Derivative(autodiff::reverse::detail::pow(x.m_expr, e));
}

/// Element-wise max used by Relu's cwiseMax – picks the larger value and
/// routes the gradient only to the selected operand.
inline Derivative max(const Derivative& a, const Derivative& b) {
    return a.val() >= b.val() ? a : b;
}
inline Derivative min(const Derivative& a, const Derivative& b) {
    return a.val() <= b.val() ? a : b;
}

// ---------------------------------------------------------------------- //
//  Stan Math-compatible API helpers
// ---------------------------------------------------------------------- //

using BaseType = double;   ///< Underlying scalar of Derivative

/**
 * @brief Retrieves the gradient from a Derivative after backprop
 */
inline BaseType getGradient(const Derivative& derivative) {
    return derivative.adj();
}

/**
 * @brief RAII guard that scopes a forward+backward pass.
 *
 * With the autodiff backend gradients are reset inside Derivative::grad(),
 * so this guard does not need to manage memory.  It is retained for API
 * compatibility with existing code.
 */
struct GradientGuard {
    GradientGuard()  = default;
    ~GradientGuard() = default;
};

} // namespace neural

// ---------------------------------------------------------------------- //
//  Eigen integration for neural::Derivative
// ---------------------------------------------------------------------- //

#include <Eigen/Core>

namespace Eigen {

template<>
struct NumTraits<neural::Derivative> : NumTraits<double>
{
    typedef neural::Derivative Real;
    typedef neural::Derivative NonInteger;
    typedef neural::Derivative Nested;

    enum {
        IsComplex             = 0,
        IsInteger             = 0,
        IsSigned              = 1,
        RequireInitialization = 1,
        ReadCost              = 1,
        AddCost               = 3,
        MulCost               = 3
    };

    static neural::Derivative epsilon()         { return neural::Derivative(std::numeric_limits<double>::epsilon()); }
    static neural::Derivative dummy_precision()  { return neural::Derivative(1e-5); }
    static neural::Derivative highest()          { return neural::Derivative(std::numeric_limits<double>::max()); }
    static neural::Derivative lowest()           { return neural::Derivative(std::numeric_limits<double>::lowest()); }
    static int digits10()                        { return NumTraits<double>::digits10(); }
};

// Allow mixed Derivative/double operations in Eigen expressions
template<typename BinOp>
struct ScalarBinaryOpTraits<neural::Derivative, double, BinOp> {
    typedef neural::Derivative ReturnType;
};

template<typename BinOp>
struct ScalarBinaryOpTraits<double, neural::Derivative, BinOp> {
    typedef neural::Derivative ReturnType;
};

} // namespace Eigen

#else // AUTO_DIFF_ENABLED not set

// Inference-only mode – autodiff/gradients are not available.
// Forward declarations to keep compiler happy (these can never be called
// due to the std::enable_if guards in the layer/optimizer headers).
namespace neural {
    using Derivative = void;
    void getGradient();
}

#endif // AUTO_DIFF_ENABLED
#endif // NEURAL_GRADIENT_HPP
