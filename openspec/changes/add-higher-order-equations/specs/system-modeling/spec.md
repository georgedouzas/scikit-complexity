# Spec Delta

## ADDED Requirements

### Requirement: States declare an order with initial derivatives
A state SHALL declare an order `k`, and the framework SHALL add companion states for its derivatives of orders 1 to
`k - 1`. These SHALL take initial values in the constructor, appear in trajectories by path, and be readable in laws.

#### Scenario: Initial angular velocity
- **WHEN** a pendulum declares its angle with order 2 and is created with `theta=0.1, theta_d1=0.5`
- **THEN** the run starts with angle 0.1 and angular velocity 0.5, and the trajectory returns both `bob.theta` and
  `bob.theta_d1`

#### Scenario: First order unchanged
- **WHEN** a state declares no order
- **THEN** the model compiles exactly as before, with no companion states

### Requirement: Laws of any order superpose on the highest derivative
Rate and noise laws on a state of order `k` SHALL contribute to its `k`-th derivative, and the contributions of
different laws SHALL add up. A law that asserts a different order SHALL raise a named error.

#### Scenario: Newton's second law from separate forces
- **WHEN** a mass on a spring receives a restoring law from one interaction and a damping law that reads `x_d1` from
  another, on a position of order 2
- **THEN** the run matches the damped oscillator `m x'' = -k x - c x'` within the solver tolerance

#### Scenario: Third order
- **WHEN** Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1` is written as one rate law on a state of order 3
- **THEN** it compiles to three first-order equations, and with A = 0.6 its trajectory is bounded and chaotic

#### Scenario: Order mismatch
- **WHEN** a law asserts order 1 on a state declared with order 2
- **THEN** compiling raises a named error that names the law, the state, and both orders

### Requirement: Equations state implicit relations in the highest derivatives
An equation law SHALL relate the highest derivatives of states to other expressions. The compiler SHALL solve all such
equations of a system jointly for the highest derivatives, and SHALL raise a named error when they cannot be solved.

#### Scenario: Pendulum as written
- **WHEN** a pendulum states `m l derivative(θ, 2) == -m g sin θ`
- **THEN** the run equals the one from the rate law `derivative(θ, 2) = -(g / l) sin θ`

#### Scenario: Coupled equations
- **WHEN** two equations each contain the second derivatives of both angles of a double pendulum
- **THEN** the compiler solves them jointly, and the energy stays constant within the solver tolerance

#### Scenario: Unsolvable highest derivative
- **WHEN** an equation contains the square of a highest derivative
- **THEN** compiling raises a named error that names the equation and the derivative

### Requirement: Difference equations take any number of lags
A state of a discrete map SHALL declare how many past values it keeps. Update laws SHALL read them as lags, and the
initial values of the lags SHALL be given in the constructor.

#### Scenario: Multiplier-accelerator
- **WHEN** Samuelson's model `Y_t = c (1 + v) Y_{t-1} - c v Y_{t-2} + G` runs with c = 0.8 and v = 1.5
- **THEN** income oscillates with growing amplitude, as the complex characteristic roots of modulus `sqrt(c v) > 1`
  predict

#### Scenario: Fibonacci recurrence
- **WHEN** `x_t = x_{t-1} + x_{t-2}` runs from 1 and 1
- **THEN** the trajectory is the Fibonacci sequence
