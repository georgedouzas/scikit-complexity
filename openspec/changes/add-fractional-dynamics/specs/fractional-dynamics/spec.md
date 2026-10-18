# Spec Delta

## Purpose

Lets a user write fractional-order differential equations, solve them with memory-aware schemes, and check them
against Mittag-Leffler solutions.

## ADDED Requirements

### Requirement: Fractional rate laws declare Caputo derivatives
A law SHALL declare that the Caputo derivative of order `0 < α ≤ 1` of a state equals an expression, and order 1 SHALL
give the ordinary rate law.

#### Scenario: Order one
- **WHEN** a fractional relaxation runs with order 1
- **THEN** its trajectory matches the exponential decay within the solver tolerance

### Requirement: Fractional solvers converge to known solutions
The framework SHALL solve fractional systems with Grünwald-Letnikov, L1, and predictor-corrector schemes, with full or
windowed memory.

#### Scenario: Mittag-Leffler relaxation
- **WHEN** `D^0.6 x = -x` runs from `x = 1` with each scheme and steps 0.01 and 0.005
- **THEN** each scheme matches `E_0.6(-t^0.6)` at time 5, and its error shrinks with the smaller step

#### Scenario: Short memory
- **WHEN** the same model runs with a memory window of 10 time units
- **THEN** the result stays within 1e-3 of the full-memory solution at time 20

### Requirement: The Mittag-Leffler function is available
The analysis layer SHALL evaluate the two-parameter Mittag-Leffler function and return it as the exact solution of
linear fractional relaxation.

#### Scenario: Exponential case
- **WHEN** a user evaluates `E_{1,1}(z)`
- **THEN** it equals `exp(z)` within 1e-12
