# stochastic-maps Specification

## Purpose
Lets a user write discrete-time stochastic processes as update laws that draw from the simulation's random stream,
including Markov chains and hidden Markov sources over finite alphabets, and run them reproducibly and compiled.

## Requirements

### Requirement: Update laws draw random values
The modelling layer SHALL provide uniform, normal, Bernoulli, and categorical draws as symbolic expressions that update
laws use like any other expression. The arguments of a draw MAY be expressions of states and parameters, and SHALL take
their values from before the step. A draw SHALL take a fresh value at every step, independently for every member of a
group.

#### Scenario: Uniform draw statistics
- **WHEN** an element with the update `x ← uniform('u', 0, 1)` runs for 100000 steps
- **THEN** the mean of `x` is 0.5 within 0.005 and its lag-1 autocorrelation is below 0.01 in absolute value

#### Scenario: Independent members
- **WHEN** a group of 1000 members has the update `x ← x + normal('step', 0, 1)` and runs for 100 steps from 0
- **THEN** the variance of `x` over the members is 100 within 15%, and the correlation of the final values of the first
  500 members with those of the last 500 is 0 within 0.1

#### Scenario: State-dependent arguments
- **WHEN** the Bernoulli probability of a draw is the expression `p * x` of a state `x` and a parameter `p`
- **THEN** each step uses the values of `x` and `p` before the step

### Requirement: Draws are shared by name
Every draw SHALL carry a name. Every use of the same name in the same component SHALL read the same value within one
step, and draws with different names SHALL be independent. Two uses of one name with different distributions or
arguments SHALL be rejected when compiling.

#### Scenario: Shared draw within a step
- **WHEN** an element has the updates `a ← bernoulli('flip', 0.5)` and `b ← 1 - bernoulli('flip', 0.5)`
- **THEN** `a + b` equals 1 at every frame

#### Scenario: Conflicting draw
- **WHEN** one component uses the name `u` for `uniform('u', 0, 1)` and for `uniform('u', 0, 2)`
- **THEN** compiling raises a named error that names the component, the draw name, and both declarations

### Requirement: Draws belong to update laws
Compiling a system SHALL accept draws only in update laws. A draw in a rate, noise, jump, boundary, equation, or event
law SHALL raise a named error that names the law and the draw and points to noise and jump laws for randomness in
continuous time.

#### Scenario: Draw in a rate law
- **WHEN** a rate law contains a normal draw
- **THEN** compiling raises a named error that names the rate law and the draw, and suggests a noise law

### Requirement: Transition laws give discrete-time Markov chains
A transition law SHALL move a state over a finite alphabet, a categorical domain or the integers from 0 to `n - 1`, to
the value drawn from the row of a transition matrix that the state's current value selects. Matrix entries SHALL be
numbers or expressions of parameters, and SHALL change live with them. A transition law SHALL be a kind of update law,
so it combines with other update laws in the discrete map form.

#### Scenario: Stationary distribution
- **WHEN** a two-symbol chain with transition matrix `[[0.9, 0.1], [0.4, 0.6]]` runs for 100000 steps
- **THEN** the fraction of steps in the first symbol is 0.8 within 0.01

#### Scenario: Live matrix change
- **WHEN** a running chain's stay probability parameter changes from 0.9 to 0.5
- **THEN** the frames after the change follow the new matrix, and the run-length distribution of the first symbol
  changes from mean 10 to mean 2

### Requirement: Emission laws give hidden Markov sources
A transition law with a source SHALL set an observed state to the value drawn from the row of its matrix that the
source state selects. It SHALL read the source state's value after the same step, so the hidden and observed sequences
are aligned frame by frame. Its matrix SHALL be symbolic, and a function matrix with a source SHALL raise a
`ValidationError`.

#### Scenario: Aligned emission
- **WHEN** a hidden chain with two states emits through a transition law with the hidden state as its source and the
  identity matrix
- **THEN** the observed sequence equals the hidden sequence at every frame

#### Scenario: Emission statistics
- **WHEN** a hidden Markov source with emission matrix `[[0.7, 0.3], [0.2, 0.8]]` runs for 100000 steps
- **THEN** the frequency of each observed symbol given each hidden state matches its emission row within 0.01

#### Scenario: Function matrix with a source
- **WHEN** a transition law with a source is given a function as its matrix
- **THEN** preparing it raises a `ValidationError` that asks for a symbolic matrix

### Requirement: Markov matrices are checked
Compiling a transition law SHALL check that its matrix is square without a source, has one column per value of the
target alphabet and one row per value of the selecting state, has nonnegative entries, and has rows that sum to one
within 1e-9, using the parameter values at compile time. Changing a parameter of a running simulation SHALL apply the
same check to every matrix that the parameter enters.

#### Scenario: Row that does not sum to one
- **WHEN** a transition matrix has a row that sums to 0.9
- **THEN** compiling raises a named error that names the law, the row, and the sum

#### Scenario: Live change breaks a row
- **WHEN** a running simulation sets a parameter so that a matrix entry becomes negative
- **THEN** the change is rejected with a named error that names the parameter, the law, and the entry, and the
  simulation keeps the previous value

### Requirement: Stochastic maps are reproducible
The discrete map stepper SHALL draw from the simulation's random generator, so two runs with the same random state
SHALL give the same trajectory. Within a step, draws SHALL be taken in a fixed order, by component path, then draw name,
then member index, so the trajectory does not depend on the order of the laws.

#### Scenario: Same seed
- **WHEN** a hidden Markov source runs twice with random state 0
- **THEN** the two trajectories are equal

#### Scenario: Law order does not matter
- **WHEN** the same system is built with its update laws in two different orders and runs with random state 0
- **THEN** the two trajectories are equal

### Requirement: Stochastic maps run compiled on every backend
Stochastic maps SHALL compile on the NumPy and Numba backends. The two backends SHALL give equal trajectories for
uniform, Bernoulli, categorical, transition, and emission draws from the same random state, and the same distribution
for normal draws.

#### Scenario: Backends agree
- **WHEN** a chaos-game iterated function system runs for 1000 steps with random state 0 on the NumPy and the Numba
  backends
- **THEN** the two trajectories are equal within 1e-12

#### Scenario: Compiled speed
- **WHEN** a group of 100000 independent two-state chains runs for 1000 steps on the Numba backend
- **THEN** the run takes less than 10 times as long as the same group with a deterministic update of the same size

### Requirement: Discrete draws take a continuous relaxation
`bernoulli` and `categorical` SHALL take `temperature`, a positive number or `None`. With a temperature, the JAX
lowering SHALL replace the draw by its concrete relaxation of the uniform value of the step (Maddison et al. 2017, Jang
et al. 2017): `sigmoid((logit p - logit u) / temperature)` for a Bernoulli draw, and `sum_k sigmoid((u - c_k / W) / temperature)`
over the cumulative weights `c_k` before the last and the total weight `W` for a categorical draw, the relaxation of
its inverse distribution function of the same uniform value.

#### Scenario: Gradient through a relaxed Bernoulli draw
- **WHEN** the mean over fixed uniform values of a relaxed Bernoulli draw of probability `p` at temperature 0.5 is
  differentiated in `p` on JAX
- **THEN** the gradient agrees with central finite differences of step 1e-6 within 1e-6 relative error

#### Scenario: Relaxed draws converge to exact draws
- **WHEN** a relaxed Bernoulli draw of probability 0.3 is lowered on JAX at temperature 0.01 for 20000 uniform values
- **THEN** the mean of the values lies within 0.02 of 0.3

### Requirement: Relaxed draws take straight-through hard values
`bernoulli` and `categorical` SHALL take `hard`. With `hard=True` and a temperature, the JAX lowering SHALL give the exact
draw in the forward pass and the gradient of the relaxation, as a straight-through estimator.

#### Scenario: Hard values
- **WHEN** a relaxed Bernoulli draw with `hard=True` is lowered on JAX for 1000 uniform values
- **THEN** every value is 0 or 1 and the gradient in `p` is the gradient of the relaxation

### Requirement: Other backends draw relaxed draws exactly
Every backend other than JAX SHALL draw a Bernoulli or categorical draw exactly and ignore its relaxation.

#### Scenario: Exact draws on NumPy
- **WHEN** a map with a relaxed Bernoulli draw runs on the NumPy backend
- **THEN** every drawn value is 0 or 1

### Requirement: Discrete draws without a relaxation are not differentiable
The JAX lowering SHALL report a `not-differentiable` warning that names a Bernoulli or categorical draw without a
temperature, and differentiable runs SHALL stop at it.

#### Scenario: Discrete draw without a temperature
- **WHEN** a system whose update draws `bernoulli('flip', p)` without a temperature is lowered on JAX for a
  differentiable run
- **THEN** the report holds a `not-differentiable` issue that names the draw `flip` and asks for a temperature
