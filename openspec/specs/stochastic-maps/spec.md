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
An emission law SHALL set an observed state to the value drawn from the row of an emission matrix that a hidden state
selects. The emission SHALL read the hidden state's value after the same step, so the hidden and observed sequences
are aligned frame by frame.

#### Scenario: Aligned emission
- **WHEN** a hidden chain with two states emits through the identity emission matrix
- **THEN** the observed sequence equals the hidden sequence at every frame

#### Scenario: Emission statistics
- **WHEN** a hidden Markov source with emission matrix `[[0.7, 0.3], [0.2, 0.8]]` runs for 100000 steps
- **THEN** the frequency of each observed symbol given each hidden state matches its emission row within 0.01

### Requirement: Markov matrices are checked
Compiling a transition or emission law SHALL check that its matrix is square for transitions, has one column per value
of the target alphabet and one row per value of the selecting state, has nonnegative entries, and has rows that sum to
one within 1e-9, using the parameter values at compile time. Changing a parameter of a running simulation SHALL apply
the same check to every matrix that the parameter enters.

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
