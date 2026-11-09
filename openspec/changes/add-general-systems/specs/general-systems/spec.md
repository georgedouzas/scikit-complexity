# Spec Delta

## Purpose

Defines the general systems of `skcomplex.systems.general` that systems of several fields share, what each takes as
data, what parts it builds, and how a field system subclasses it with its own parameters.

## ADDED Requirements

### Requirement: General systems are recipes that several fields share
A general system SHALL serve systems of two fields, or be able to, and SHALL be named after its mathematics, never
after a field quantity. As a recipe of several fields, by step 3 of the placement rule, it SHALL live in a concept
subpackage of `skcomplex.systems.general`, here `skcomplex.systems.general.dynamics`, and no core module SHALL import
it.

#### Scenario: Import a general system
- **WHEN** a user imports `DynamicalSystem`, `NetworkDynamicalSystem`, `ReactionDiffusionSystem`, or
  `StockFlowSystem` from `skcomplex.systems.general.dynamics`
- **THEN** the import succeeds, and the class is a subclass of `BaseSystem` from `skcomplex.core.components.systems`

#### Scenario: The core holds none of them
- **WHEN** a user imports `DynamicalSystem` from `skcomplex.core.components.systems`
- **THEN** the import fails with `ImportError`, and no module under `src/skcomplex/core` imports
  `skcomplex.systems.general.dynamics`

#### Scenario: Shape of one field
- **WHEN** a system shape appears in one field only, such as the Boolean networks of systems biology
- **THEN** it stays in the `common` package of that field, and no general system holds it

#### Scenario: Shape covered by one core part
- **WHEN** a system shape is one core interaction on a group, such as a cellular automaton, a `CompositeSystem` with a
  `LocalRuleInteraction`
- **THEN** no general system duplicates it

### Requirement: Dynamical systems are given by states and equations
`DynamicalSystem` SHALL take the initial value of every state, the equation of every state, and the parameters, all
by name, and SHALL build one element that holds the states and the parameters under the label it is given. An equation
SHALL be a function of a namespace whose `target` holds the states and the parameters and whose `t` holds the time.
The trajectory SHALL name every state `<element>.<state>`.

#### Scenario: Exponential decay
- **WHEN** a user simulates `DynamicalSystem(states={'x': 1.0}, equations={'x': lambda s: -s.target.k * s.target.x},
  parameters={'k': 0.5}, element='tank')` until 2
- **THEN** the trajectory holds `tank.x`, and its last value is `exp(-1)` within the tolerance of the solver

#### Scenario: Unknown state
- **WHEN** a user prepares a `DynamicalSystem` with an equation for a state it does not declare
- **THEN** preparing raises a `ValidationError` that names the state and the declared states

### Requirement: Dynamical systems run in continuous or discrete time
`DynamicalSystem` SHALL read its equations as rates in the time `'continuous'` and as the next values in the time
`'discrete'`. In continuous time it SHALL take optional noise amplitudes by state, read in the Itô or Stratonovich
convention. Its tags SHALL follow from the parts it builds, with no class chosen by the form of the equations.

#### Scenario: Map of the logistic family
- **WHEN** a user simulates `DynamicalSystem(states={'x': 0.2}, equations={'x': lambda s: s.target.r * s.target.x *
  (1 - s.target.x)}, parameters={'r': 2.0}, time='discrete')` for 30 steps
- **THEN** the last value of `x` is `0.5` within 1e-9, and the tags name the law `'update'` and a discrete time

#### Scenario: Ornstein-Uhlenbeck process
- **WHEN** a user gives a rate `-x` and the noise amplitude `0.3` for `x` and reads the tags
- **THEN** the tags name the laws `'rate'` and `'noise'` and the randomness `'stochastic'`

#### Scenario: Noise in discrete time
- **WHEN** a user prepares a `DynamicalSystem` in discrete time with a noise amplitude
- **THEN** preparing raises a `ValidationError` that says noise amplitudes need continuous time

### Requirement: Maps of one step read the states at its start
In discrete time every equation of a `DynamicalSystem` SHALL read the values of the states at the start of the step,
so the order of the states never changes the result.

#### Scenario: Hénon map
- **WHEN** a user builds the Hénon map with `x' = 1 - a x^2 + y` and `y' = b x`, once with the states in the order
  `x`, `y` and once in the order `y`, `x`, and simulates both for 10 steps
- **THEN** both trajectories are equal

### Requirement: Network dynamical systems couple the nodes of a graph
`NetworkDynamicalSystem` SHALL put the states of a `DynamicalSystem` on every node of a graph it is given and SHALL
add, for every state it has a coupling for, the sum over the neighbours `sum_j A_ij g(x_i, x_j)` to its equation. The
graph SHALL be required, and the initial values SHALL be scalars for every node or arrays with one value per node.

#### Scenario: Diffusion on a path
- **WHEN** a user simulates a `NetworkDynamicalSystem` on a path of three nodes with no local rate, the coupling
  `D (x_j - x_i)`, and the initial values `[1, 0, 0]`, for a long time
- **THEN** every node tends to `1/3`, and the total stays `1` within the tolerance of the solver

#### Scenario: Missing graph
- **WHEN** a user prepares a `NetworkDynamicalSystem` without a graph
- **THEN** preparing raises a `ValidationError` that asks for a graph

### Requirement: Reaction-diffusion systems diffuse fields on a lattice
`ReactionDiffusionSystem` SHALL put the states of a `DynamicalSystem` as fields on the sites of a lattice of a given
shape, spacing, and boundary, with the equations as local kinetics. It SHALL add `D laplacian(u)` to the rate of every
field `u` it has a diffusivity `D` for, as an interaction apart from the kinetics, so a split solver can treat it.

#### Scenario: Fisher-Kolmogorov front
- **WHEN** a user simulates a one-dimensional `ReactionDiffusionSystem` with the kinetics `r u (1 - u)`, the
  diffusivity `D`, and a step of `u` at the left end
- **THEN** the front moves to the right at a speed within 5 percent of `2 sqrt(r D)`

#### Scenario: Field with no diffusivity
- **WHEN** a field has kinetics and no diffusivity
- **THEN** it changes only by its kinetics, site by site

### Requirement: Stock-flow systems move stocks through named flows
`StockFlowSystem` SHALL take the initial value of every stock and every flow by name, each flow with the stock it
leaves or `None` for a source, the stock it enters or `None` for a sink, and its rate as a function of a namespace
whose `source` holds the stocks and the parameters. Every stock SHALL change only through its flows, and the trajectory
SHALL report every flow as `<element>.<flow>`.

#### Scenario: Bathtub
- **WHEN** a user simulates a `StockFlowSystem` with the stock `water`, the flow `faucet` from a source into `water`,
  and the flow `drain` from `water` to a sink, at equal rates
- **THEN** the water stays constant, and the trajectory holds `tub.faucet` and `tub.drain`

#### Scenario: Flow of an unknown stock
- **WHEN** a flow names a stock the system does not declare
- **THEN** preparing raises a `ValidationError` that names the flow and the stock

### Requirement: Field systems subclass a general system through hooks
A field system SHALL subclass a general system by taking the parameters of its model under its own names and by
overriding the hooks that return the states, the equations or flows, the parameters, the declared variables with
their specs, and the label of the element. The general system SHALL read its data only through these hooks.

#### Scenario: Field system with its own parameters
- **WHEN** a field system on `DynamicalSystem` takes `sigma`, `rho`, and `beta` and returns the Lorenz rates from its
  hook of equations
- **THEN** `get_params()` lists `sigma`, `rho`, `beta`, and `label`, and `set_params(rho=10.0)` changes the next
  `prepare`

#### Scenario: Declared domains
- **WHEN** a field system on `StockFlowSystem` returns stocks declared nonnegative from its hook of declared variables
- **THEN** a negative initial stock raises the same `ValidationError` as an element with that domain
