# group-reductions Specification

## Purpose
Lets a law put a reduction over the members of a group or a set of elements, or an observable of its system, inside
any expression, and evaluates it at the current state on every backend.

## Requirements

### Requirement: Laws reduce over groups inside any expression
The modelling layer SHALL provide the symbolic reductions total, mean, maximum, minimum, variance, and count over the
members of a group or a set of elements, usable inside any law expression, where the symbols inside a reduction stand
for the reduced member and the symbols outside it stand for the reading member.

#### Scenario: Nonlinear function of a total
- **WHEN** every member of a group of 50 follows `x' = -x + tanh(total(x))` and the drift is evaluated at a random
  state
- **THEN** each member's rate equals `-x_i + tanh(sum_j x_j)` within 1e-12

#### Scenario: Instantaneous mean in an interaction
- **WHEN** an interaction whose role binds to a group of theta neurons adds `kappa * mean(a_n * (1 - cos(theta))^n)`
  to every neuron's rate
- **THEN** the drift at a random state equals the formula with the mean of the members' pulses at that state within
  1e-12, and the system has no state besides the phases

#### Scenario: Maximum drives a single element
- **WHEN** an inhibitor's rate is `phi * (heaviside(maximum(x) - theta_z) - z)` over a group of oscillators
- **THEN** the drive is 1 exactly when at least one oscillator's `x` exceeds `theta_z`, and 0 otherwise

#### Scenario: Count with a condition
- **WHEN** a law reads `count(where=x > 0.5)` over a group of 10 members, 3 of which exceed 0.5
- **THEN** the law reads 3, and a mean with the same condition reads the mean of those 3 members

#### Scenario: No selected members
- **WHEN** a condition selects no member
- **THEN** total, mean, maximum, minimum, and variance read 0 and count reads 0

### Requirement: Reductions take constant weights
A total or a mean SHALL accept a constant weight matrix, dense or sparse, with one row per reading member and one
column per reduced member, so the reading member `i` reads `sum_j W_ij a_j`.

#### Scenario: Payoff matrix
- **WHEN** a group of 3 strategies reads `total(x, weights=A)` with a 3 by 3 payoff matrix `A`
- **THEN** member `i` reads `(A x)_i` within 1e-12, with `A` as a NumPy array and as a SciPy sparse matrix

#### Scenario: Weights of the wrong shape
- **WHEN** the weight matrix of a reduction over a group of 3 has shape `(3, 4)`
- **THEN** compiling raises a named error that names the reduction, the group, and both shapes

### Requirement: Reductions nest
The argument of a reduction SHALL be allowed to hold other reductions, and the compiler SHALL evaluate them in
dependency order.

#### Scenario: Mean fitness
- **WHEN** replicator members read `phi = total(x * total(x, weights=A))` inside `x' = x * (total(x, weights=A) - phi)`
- **THEN** the rates equal `x_i ((A x)_i - x^T A x)` within 1e-12, and their sum is 0 within 1e-12 on the simplex

### Requirement: Reductions name one group
A reduction SHALL reduce over the one group or element set whose symbols its argument reads, and the compiler SHALL
reject an argument that reads two groups or a reduction whose group does not exist.

#### Scenario: Two groups in one reduction
- **WHEN** the argument of a total reads states of the groups `prey` and `predators`
- **THEN** compiling raises a named error that names the reduction and both groups and asks for one reduction per
  group

### Requirement: Laws read observables at the current state
A law SHALL read a symbolic observable of its own partition at the state of the evaluation, and an observable of
another partition at the last synchronisation point, and a Python callable observable SHALL be readable only from
another partition.

#### Scenario: Kuramoto through observables
- **WHEN** a system declares the observables `X = mean(cos theta)` and `Y = mean(sin theta)` over its oscillators and
  each oscillator's rate is `omega + K (Y cos theta - X sin theta)`
- **THEN** the system compiles to one ODE partition, and its drift equals the all-to-all sum
  `omega_i + (K / N) sum_j sin(theta_j - theta_i)` within 1e-12

#### Scenario: Callable observable in the same partition
- **WHEN** a law reads an observable defined by a Python function in its own partition
- **THEN** compiling raises a named error that names the observable and tells the user to write it symbolically or
  move the reading law to another partition

### Requirement: Reductions follow the law kind
Reductions SHALL be evaluated at the state each law kind reads: the current state for drift, noise, boundary, and event
laws, the state at step `n` for update laws, and the state after the last event for jump propensities.

#### Scenario: Synchronous map
- **WHEN** a group of coupled logistic maps updates as `x' = (1 - eps) f(x) + eps * mean(f(x))`
- **THEN** every member's next value uses the mean of the values at step `n`, not values already updated

#### Scenario: Propensity after an event
- **WHEN** a mean-field SIS jump process has infection propensities proportional to `count(where=infected)`
- **THEN** after every event the propensities equal those computed from the new state

### Requirement: Backends agree on reductions at linear cost
The NumPy and Numba backends SHALL produce the same results for laws with reductions, and an evaluation SHALL cost
time linear in the group size.

#### Scenario: Backends agree
- **WHEN** a mean-field Kuramoto group of 2000 members is compiled with both backends and evaluated on the same state
- **THEN** the drifts agree within 1e-12 relative error

#### Scenario: Linear cost
- **WHEN** one drift evaluation of a mean-field Kuramoto group is timed at 10000 and 100000 members on Numba after
  compilation
- **THEN** the larger group takes at most 15 times as long as the smaller

### Requirement: Reductions have an exact Jacobian
The compiled problem SHALL derive the Jacobian of laws with reductions by the chain rule through each reduction, and
a sparsity pattern, when the problem gives one, SHALL include every reduced member that a reading member depends on.

#### Scenario: Jacobian against finite differences
- **WHEN** the exact Jacobian of a group of 20 members with `x' = -x + tanh(total(x)) + mean(x^2)` is evaluated at a
  random state
- **THEN** it equals a central finite-difference Jacobian within 1e-6, and its pattern covers every nonzero entry
