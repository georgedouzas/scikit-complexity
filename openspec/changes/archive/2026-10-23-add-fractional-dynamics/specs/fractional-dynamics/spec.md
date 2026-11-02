# Spec Delta

## Purpose

Lets a user write fractional-order differential equations as rate laws on states of fractional order, solve them with
memory-aware schemes, and check them against Mittag-Leffler solutions.

## ADDED Requirements

### Requirement: States take fractional orders
A state SHALL accept an order `0 < α < 1`, and the rate laws that target it SHALL declare that its Caputo derivative of
order `α` equals the sum of their expressions. Order 1 SHALL give the ordinary rate law, and no law kind SHALL be
specific to fractional states.

#### Scenario: Order one
- **WHEN** a fractional relaxation runs with order 1
- **THEN** its trajectory matches the exponential decay within the solver tolerance

#### Scenario: Rate law on a fractional state
- **WHEN** an element declares `x` with `state(order=0.6)` and the law `Rate(x, -x)`
- **THEN** the system compiles to the fractional form without any fractional law

### Requirement: Fractional orders are validated
Solvers and backends SHALL declare whether they support fractional orders. Validation SHALL report an order that is
neither an integer nor in `(0, 1)`, a fractional state under a time that is not continuous, and a fractional state with
a solver or backend that does not support fractional orders, each with the state's path and a fix.

#### Scenario: Solver without fractional orders
- **WHEN** a system with a fractional state is validated with the solver `'rk4'`
- **THEN** the report holds one error that names the state and lists the fractional solvers

#### Scenario: Order out of range
- **WHEN** a state declares order 1.5
- **THEN** the report holds one error that names the state and the accepted interval

#### Scenario: Fractional state under discrete time
- **WHEN** a system declared with discrete time holds a fractional state
- **THEN** the report holds one error that names the state and suggests continuous time

### Requirement: Fractional solvers converge to known solutions
The framework SHALL solve fractional systems with Grünwald-Letnikov, L1, and predictor-corrector schemes, with full or
windowed memory, and each scheme SHALL declare its support for fractional orders.

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
