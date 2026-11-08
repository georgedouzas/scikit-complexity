## Purpose

Lets a user count the fixed points of large flows and maps by stability index, also over many quenched realizations
of a random system, and compare the typical count with the average count.

## ADDED Requirements

### Requirement: Fixed points are searched numerically with deflation
`stability(kind='census')` SHALL search the fixed points of an ODE system or a map from a given number of starting
points drawn uniformly within given bounds, with a seed. It SHALL deflate every root it has found, so that a later
search does not return it again, and SHALL return every distinct root once.

#### Scenario: Bistable flow
- **WHEN** a user takes the census of `dx/dt = x - x^3` from 20 starting points in `[-2, 2]`
- **THEN** it returns the three fixed points -1, 0, and 1 within 1e-8, each once

#### Scenario: Many roots
- **WHEN** a user takes the census of `dx/dt = sin(x)` from 50 starting points in `[-10, 10]`
- **THEN** it returns the seven roots `k pi` for `k` from -3 to 3 within 1e-8, each once

#### Scenario: Same seed
- **WHEN** a user takes the census of one system twice with the same seed
- **THEN** both results hold the same points in the same order

### Requirement: Every fixed point has a stability index
The census SHALL give every fixed point its index, the number of eigenvalues of the Jacobian there with positive real
part for a flow and with modulus above one for a map, and SHALL count the points by index from 0 to the number of
state components.

#### Scenario: Indices of the bistable flow
- **WHEN** a user takes the census of `dx/dt = x - x^3`
- **THEN** the points -1 and 1 have index 0, the point 0 has index 1, and the counts by index are `[2, 1]`

#### Scenario: Saddle of a map
- **WHEN** a user takes the census of the linear map `x' = 2 x, y' = y / 2`
- **THEN** it returns the origin with index 1

### Requirement: The census reads the numeric drift
The census SHALL evaluate the compiled drift or next value of a system and its Jacobian numerically, from the exact
Jacobian of the laws when it exists and by finite differences otherwise. It SHALL accept groups of any size and
callable laws, and SHALL reject other forms with a named error.

#### Scenario: Group above the symbolic limit
- **WHEN** a user takes the census of a group of 200 uncoupled members that each relax to 0
- **THEN** it returns the origin alone, with index 0, without the error of the 64-member limit

#### Scenario: Callable law
- **WHEN** a user takes the census of a system whose rate law calls a NumPy function of the state
- **THEN** it returns its fixed points and their indices

#### Scenario: Stochastic system
- **WHEN** a user takes the census of an SDE system
- **THEN** an `AnalysisError` states that the census reads ODE systems and maps

### Requirement: The census averages over quenched realizations
With `realizations` above one, the census SHALL redraw the disorder of a system that takes a `random_state`, with
seeds derived from its own seed, and SHALL return the counts by index of every realization, the annealed count, the
mean count, and the quenched count, `exp(mean log(1 + n)) - 1`, with their logarithms divided by the number of state
components as the annealed and quenched complexities.

#### Scenario: Trivial phase
- **WHEN** a user takes the census of the random force field relaxation system at a rescaled disorder of 0.5 over 10
  realizations
- **THEN** every realization has exactly one fixed point, the origin, with index 0, and both complexities are 0

#### Scenario: Typical count below the mean
- **WHEN** a user takes the census of the random force field relaxation system at a rescaled disorder of 3 and a
  reciprocity of 1 over 20 realizations
- **THEN** the mean count exceeds 1 and the quenched count does not exceed the mean count

#### Scenario: System without disorder
- **WHEN** a user asks for 10 realizations of a system that takes no `random_state`
- **THEN** an `AnalysisError` states that realizations need a system with a `random_state`
