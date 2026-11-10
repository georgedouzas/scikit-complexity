## MODIFIED Requirements

### Requirement: Bilinear laws are one law
One bilinear law SHALL express gains and losses of a density at the rate of the product of two densities. A conserved
bilinear law SHALL also take the matching loss from its first density, so the total over its classes is unchanged, and
a law without a second density SHALL redistribute the first linearly.

#### Scenario: Encounter outcomes
- **WHEN** an encounter is destructive or proliferative
- **THEN** it lowers to the bilinear law with the sign of its outcome, with the trajectories it had before

#### Scenario: Conservative encounter
- **WHEN** an encounter is conservative
- **THEN** it lowers to a conserved bilinear law, with the trajectories it had before

#### Scenario: Linear redistribution
- **WHEN** a conserved bilinear law with no second density moves a density of two classes from the first class to the
  second at rate 1, starting from all in the first class
- **THEN** the first class decays as `exp(-t)` within 1e-6, and the total stays constant within 1e-12

### Requirement: Law utilities
The law namespace SHALL offer `hill`, `logistic`, `pulse_train` with a count, `clip`, `minimum`, `maximum`, `sign`, the
distributed delay of any order with or without conservation, a series of any value through given points, reductions
keyed by a state, and a circular mean. It SHALL offer one function for one operation, so it SHALL NOT offer a function
that only fixes arguments of another.

#### Scenario: Hill function
- **WHEN** a rate reads `hill(x, K, n)` at `x = K`
- **THEN** it evaluates to one half

#### Scenario: Third-order material delay
- **WHEN** a rate reads `distributed_delay(x, T, 3)` of a step input
- **THEN** its output follows the third-order material delay of the step

#### Scenario: Information smoothing
- **WHEN** a rate reads `distributed_delay(x, T, 1, conserved=False)`
- **THEN** its output is the first-order exponential smoothing of `x` with time `T`

#### Scenario: Series of a state
- **WHEN** a rate reads `series(x, [0, 1, 2], [0.0, 2.0, 3.0])` at `x = 0.5` and at `x = 5`
- **THEN** it evaluates to 1.0 and to 3.0

## ADDED Requirements

### Requirement: The law namespace holds only what laws are written with
The public law namespace SHALL re-export only the functions and symbols a law is written with. The helpers that only
the compiler reads, such as the functions that list the reductions, draws, or operators of an expression, and the
constants of the compiler, SHALL NOT be part of it.

#### Scenario: Compiler helper
- **WHEN** a user imports `reductions_in` from the public law namespace
- **THEN** the import raises an `ImportError`

#### Scenario: Operator constructors stay
- **WHEN** a user writes `J * operator_product(a, b) + dagger(b)` from the public law namespace
- **THEN** the expression is a valid operator of a generator law

### Requirement: Draw functions have one import path
The draw functions `uniform`, `normal`, `bernoulli`, `categorical`, and `draw` SHALL be importable from the public law
namespace alone, and the module that defines them SHALL be a module of that namespace.

#### Scenario: One path
- **WHEN** a user imports `uniform` from the public law namespace
- **THEN** the import succeeds, and no other public package of the core re-exports `uniform`
