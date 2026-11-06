## RENAMED Requirements

- FROM: `### Requirement: Parts share three verbs`
- TO: `### Requirement: Parts share two verbs`

## MODIFIED Requirements

### Requirement: Parts share two verbs
Elements, interactions, laws, and systems SHALL derive from `BasePart`, which defines `label`,
`prepare(namespace)`, and `run(input)`. Elements, interactions, and systems SHALL return `self` from
`prepare`, and their laws from `run(s)`; a system SHALL return the laws of its children. A law SHALL return a prepared
copy, and SHALL evaluate its contribution from `run(values)`.

#### Scenario: System yields laws
- **WHEN** a system of a population element and a growth interaction is prepared and run with a namespace
- **THEN** it returns the laws of the element and of the interaction, in order

#### Scenario: Law run alone
- **WHEN** a user prepares `RateLaw(s.n, r * s.n)` against a namespace with `n` and `r`, and runs it at `n=10`,
  `r=0.5`
- **THEN** the result is `5.0`, its `targets_` holds `n`, and its `reads_` holds `n` and `r`

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
- **THEN** `prepare` raises `ValidationError` whose message names the unpaired coordinate

#### Scenario: Coordinate of order 2
- **WHEN** a `HamiltonianInteraction` names a coordinate declared with `StateSpec(order=2)`
- **THEN** the compiler's report holds one error at the path of the interaction that names the state and gives the fix
  `StateSpec(order=1)` or `EulerLagrangeInteraction`
