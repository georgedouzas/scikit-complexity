# Spec Delta

## ADDED Requirements

### Requirement: Hamiltonian interactions generate rate laws
`HamiltonianInteraction` SHALL derive Hamilton's equations symbolically from a function of canonical coordinates and
momenta, states of order 1 that are real or angles, as one `RateLaw` per coordinate and per momentum. The Hamiltonians
of separate interactions on one state SHALL sum. A misuse SHALL raise `ValidationError` naming the state.

#### Scenario: Oscillator from a Hamiltonian
- **WHEN** an element with states `q` and `p` takes `HamiltonianInteraction` with `H = p^2 / (2 m) + k q^2 / 2`
- **THEN** its laws are `RateLaw(q, p / m)` and `RateLaw(p, -k q)`

#### Scenario: Coupling Hamiltonians sum
- **WHEN** two oscillators each take their own `HamiltonianInteraction`, and a third interaction binds both with the
  Hamiltonian `c (q1 - q2)^2 / 2`
- **THEN** the rate of `p1` is `-k q1 - c (q1 - q2)`, the sum of the terms of the two interactions

#### Scenario: Coordinate without a momentum
- **WHEN** a `HamiltonianInteraction` lists two coordinates and one momentum
- **THEN** `check()` raises `ValidationError` whose message names the unpaired coordinate

#### Scenario: Coordinate of order 2
- **WHEN** a `HamiltonianInteraction` names a coordinate declared with `StateSpec(order=2)`
- **THEN** the compiler's report holds one error at the path of the interaction that names the state and gives the fix
  `StateSpec(order=1)` or `EulerLagrangeInteraction`
