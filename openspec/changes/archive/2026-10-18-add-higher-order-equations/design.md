# Design

## Context

States hold one value each, rate laws add terms to first derivatives, update laws give next values, and the compiler
collects laws per target into the problem's functions. Sage represents expressions and solves equations. Velocity
Verlet recognises second-order structure only from rate laws of the form `dx/dt = v`.

## Goals / Non-Goals

**Goals:**

- Any differential or difference equation of any order, in any field, written as the user writes it on paper.
- The same superposition rule for every order: contributions to the highest derivative add up.

**Non-Goals:**

- Boundary value problems, such as beams with conditions at both ends. Every equation here is an initial value
  problem.
- Differential-algebraic systems whose highest derivatives cannot be solved for, which `add-acausal-components`
  reports.
- Fractional orders, which `add-fractional-dynamics` covers.

## Decisions

### Declaring order

`state(default=..., order=k, initial=(...))` declares a state of order `k`. `initial` holds the default initial values of
the derivatives of orders 1 to `k - 1`. The component gains one constructor field per derivative, named
`<state>_d<j>`, so `Pendulum('bob', theta=0.1, theta_d1=0.0)` sets the angle and the angular velocity. The fields are
states with the parent's domain for vectors and a real domain otherwise, and are addressed by path like any state, for
example `bob.theta_d1`. The default order is 1, which adds nothing.

- Alternative considered: inferring the order from the laws. Rejected because initial values of derivatives then have
  nowhere to live in the constructor, and the order of a model should be visible where its states are declared.

### Laws of any order

A rate or noise law on a state of order `k` contributes to the `k`-th derivative. Contributions add up per state, as
they do for first-order rates. `Rate(target, value, order=None)` asserts the order when given, and the compiler raises a
named error when it differs from the declaration, so that a law written for a first-order state cannot silently drive
an acceleration. The compiler then adds `d(x_d{j-1})/dt = x_d{j}` for `j < k`, which makes the rate on `x_d{k-1}`
carry the contributions. A noise law on an order-`k` state drives `x_d{k-1}`, which is Langevin dynamics for `k = 2`.

### Derivative notation and equations

`derivative(x, j)` is a Sage function that stays symbolic. In right-hand sides, the compiler replaces it with the
companion symbol `x_d{j}` for `j < k`, and rejects `j ≥ k` there. `Equation(lhs, rhs)` states an implicit relation in
which the highest derivatives of one or more states appear. The compiler collects all equations together with the rate
laws on the same states, and checks that the relation is linear in the highest derivatives. It then solves the linear
system for them symbolically, falling back to a numeric solve per evaluation above a size threshold, and adds the
solutions as rates. Coupled equations such as a double pendulum's are solved jointly. A system whose highest
derivatives appear nonlinearly, or cannot be solved for, raises a named error that lists the equations and the
derivatives.

### Difference equations of any order

`state(default=..., lags=m, initial=(...))` on a state of a discrete map keeps its last `m` values in companion states
`<state>_lag<j>`. Initial values give the values at steps `-1` to `-m`. `lag(x, j)` reads the value `j` steps before
the current one in update laws, which give the next value from the current state, and the iteration shifts the
companions after every update. Samuelson's model `Y_{t+1} = c (1 + v) Y_t - c v Y_{t-1} + G` therefore keeps one lag:
`Update(Y, c * (1 + v) * Y - c * v * lag(Y, 1) + G)`.

### Analysis

`equations_of` rebuilds the written order: for a state of order `k` it returns `diff(x(t), t, k) == rhs` with
companions replaced by derivatives of `x(t)`. For discrete maps it returns `x(t + 1) == rhs` with `x(t - j)` for lags.
`solve_symbolically` passes single equations of any order to Maxima's `desolve` with initial values of all
derivatives, and passes systems to `desolve_system` after reduction. A result is accepted only when it contains no
unevaluated `ilt`, `laplace`, `integrate`, or unknown function of `t`. Otherwise the named error explains that no
closed form was found, and suggests numeric simulation.

### Parameter signs

`declare_domains` already gives Maxima the domains of symbols. In addition, a parameter whose domain is real but whose
current value is strictly positive or negative gets that sign, which matches the numbers the user wrote. When Maxima
still asks a sign question, the error names the expression it asks about and the parameters involved, and suggests
declaring them with `domain=Domain.POSITIVE`.

## Risks / Trade-offs

- [Value-based signs can mislead when a parameter later changes sign] → Exact solutions are symbolic results for the
  declared values, and the docstring states that the sign comes from the value when the domain is real.
- [Generated constructor fields] → They follow one naming rule, appear in `get_params`, and the user guide documents
  them next to the declaration.

## Implementation Notes

- Companion states of derivatives take the real domain, whatever the domain of their state, because a derivative of a
  positive quantity can be negative. Companions of lags take the domain of their state.
- Problems record every companion with its kind, state, and index in `Problem.companions`, which the analysis uses to
  write equations in their order.
- Equations with up to six highest derivatives are solved symbolically. Larger systems are solved per evaluation with
  `numpy.linalg.solve` through `Call` laws. Rate laws on a state that also has an equation add to the solved derivative.
- One equation of order 1 or 2 is solved with `desolve`, and one of higher order with `desolve_laplace`. Values of
  unknown functions at fixed times, such as `x(0)`, count as integration constants and do not make a solution unclosed.
- `declare_domains` takes parameter values and gives real parameters the sign of their value, replacing the sign of an
  earlier value with Maxima's `forget`.
- The `nonlinear` library module starts here with Sprott's jerk equation, and `add-analysis-toolkit` extends it.
