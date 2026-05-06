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

/**
 * @brief Leaf expression node for an independent (differentiable) variable.
 *
 * Stores the adjoint directly as a plain double member.  Because LeafExpr is
 * itself reference-counted via ExprPtr (shared_ptr<Expr<double>>), the adjoint
 * lives exactly as long as any expression node in the tree holds a reference —
 * avoiding the use-after-free that would occur with a raw double* (gradPtr).
 */
struct LeafExpr : autodiff::reverse::detail::Expr<double> {
    double adj = 0.0;

    explicit LeafExpr(double v) : autodiff::reverse::detail::Expr<double>(v) {}

    void propagate(const double& wprime) override { adj += wprime; }
    void propagatex(const ExprPtr& /*wprime*/) override {}
    void update() override {}
};

// Thread-local registry of weak_ptrs to leaf nodes.
// resetAllLeaves() zeros every live leaf's adjoint and prunes expired entries.
inline std::vector<std::weak_ptr<LeafExpr>>& leafRegistry() {
    static thread_local std::vector<std::weak_ptr<LeafExpr>> registry;
    return registry;
}

inline void registerLeaf(const std::shared_ptr<LeafExpr>& leaf) {
    leafRegistry().push_back(leaf);
}

inline void resetAllLeaves() {
    auto& reg = leafRegistry();
    std::size_t dst = 0;
    for (std::size_t i = 0; i < reg.size(); ++i) {
        if (auto sp = reg[i].lock()) {
            sp->adj = 0.0;
            reg[dst++] = std::move(reg[i]);
        }
    }
    reg.resize(dst);
}
} // namespace detail

/**
 * @brief Differentiable scalar type used when training neural networks.
 *
 * Wraps a single ExprPtr (shared_ptr to an autodiff expression node).
 * Leaves created from numeric values own a LeafExpr whose `adj` member
 * accumulates the gradient during backpropagation.  Copies share the same
 * node, so adj() always reads the same gradient regardless of which copy is
 * queried.  Intermediate results (produced by arithmetic) hold a non-leaf
 * ExprPtr; calling adj() on them is undefined behaviour.
 */
class Derivative {
public:
    detail::ExprPtr m_expr;  ///< autodiff expression node (leaf or intermediate)

    // ------------------------------------------------------------------ //
    //  Constructors
    // ------------------------------------------------------------------ //

    /// Default: leaf with value 0
    Derivative() : Derivative(0.0) {}

    /// Leaf constructed from any arithmetic type
    template<typename U,
             typename = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    Derivative(U val) {
        auto leaf = std::make_shared<detail::LeafExpr>(static_cast<double>(val));
        detail::registerLeaf(leaf);
        m_expr = std::move(leaf);
    }

    /// Copy: share expression node (and its gradient storage)
    Derivative(const Derivative& other) = default;

    // ------------------------------------------------------------------ //
    //  Assignment
    // ------------------------------------------------------------------ //

    Derivative& operator=(const Derivative& other) = default;

    /// Assign from arithmetic: replace with a fresh leaf
    template<typename U,
             typename = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    Derivative& operator=(U val) {
        auto leaf = std::make_shared<detail::LeafExpr>(static_cast<double>(val));
        detail::registerLeaf(leaf);
        m_expr = std::move(leaf);
        return *this;
    }

    // ------------------------------------------------------------------ //
    //  Value and gradient access
    // ------------------------------------------------------------------ //

    /// Current value of this node in the expression tree
    double val() const { return m_expr->val; }

    /// Accumulated adjoint – valid only on leaf nodes after grad()
    double adj() const {
        return static_cast<const detail::LeafExpr*>(m_expr.get())->adj;
    }

    /// Trigger reverse-mode backpropagation from this node.
    /// Resets all live leaf adjoints to zero first.
    void grad() const {
        detail::resetAllLeaves();
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
    explicit Derivative(detail::ExprPtr e) : m_expr(std::move(e)) {}
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
