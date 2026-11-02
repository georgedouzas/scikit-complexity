# Proposal

## Why

Models in every science are written as equations of any order, not only as first-order systems:
- Newton's second law `m l θ'' = -m g sin θ` in mechanics.
- A damped oscillator in electrical circuits.
- Jerk equations of third order in chaotic dynamics (Datseris and Parlitz 2022 ch. 1, Dilão 2023 ch. 1).
- Samuelson's multiplier-accelerator model in economics, a second-order difference equation.
- Fibonacci-type recurrences in population biology.

Today every law is first order, so users split such equations by hand into auxiliary states, and the model no longer
reads like the equation they meant. Running the pendulum also showed two faults in exact solving:
- **A false solution.** `solve_symbolically` returned unevaluated inverse Laplace transforms as if they were a
  solution.
- **Parameters without a sign.** Parameters given as plain numbers have no sign, so Maxima stops and asks instead of
  solving.

## What Changes

- Add states of any order: a state declares its order `k`, and the framework adds companion states for its
  derivatives up to order `k - 1`. They take initial values like any state, appear in trajectories and the symbol
  namespace, and laws read them.
- Add laws of any order: rate and noise laws on a state of order `k` contribute to its `k`-th derivative and add up as
  first-order rates do, so independent mechanisms superpose without knowing about each other.
- Add a derivative notation: `derivative(x, j)` in any expression denotes the `j`-th time derivative of a state, and
  equation laws state an implicit relation, such as `m l derivative(θ, 2) == -m g sin(θ)`. The compiler solves for the
  highest derivatives jointly when several equations couple them.
- Compile every higher-order model to an equivalent first-order system for the existing solvers and backends, so
  velocity Verlet keeps detecting position and velocity pairs.
- Add difference equations of any order: a state of a discrete map declares how many past values it keeps,
  `lag(x, j)` reads the value `j` steps back, and update laws give the next value from them.
- Show and solve equations in their written order: `equations_of` returns higher-order relations, and
  `solve_symbolically` solves single equations of any order directly with initial values of every derivative.
- Fix exact solving: return a solution only when it contains no unevaluated transforms or integrals, and otherwise
  raise the named error. Give Maxima the sign of every parameter from its domain or its value, and name the parameter
  and the fix when a sign is still needed.
- Add library models across fields: the nonlinear pendulum written as Newton's law, a driven damped oscillator, a
  series RLC circuit as a charge equation, Sprott's jerk equation, Samuelson's multiplier-accelerator, and a
  Fibonacci-type age-structured recurrence.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: states and laws of any order, equation laws, and difference equations of any order.
- `symbolic-analysis`: higher-order equations, exact solutions of any order, no false solutions, and parameter signs.
- `model-library`: higher-order models across fields.

## Impact

- `skcomplex.core.state` gains `order` and `initial`, and `skcomplex.model` gains `derivative`, `lag`, and `Equation`.
  The compiler gains an order-reduction stage before lowering.
- Existing first-order models are unchanged, because the default order is 1.
- `add-variational-mechanics` and `add-acausal-components` build on `Equation` and `derivative` instead of defining
  their own.
- Depends on `add-general-model-interface`.
