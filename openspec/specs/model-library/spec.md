# model-library Specification

## Purpose

Provides ready-made elements and interactions for common scientific models, built only on the public framework, so a
user can assemble a model without writing laws.

## Requirements

### Requirement: Physics models
The library SHALL provide particles with position, velocity, and mass, constant forces, linear drag, and potentials
whose forces are derived from an energy expression, including harmonic potentials, springs, and gravity.

#### Scenario: Constant acceleration
- **WHEN** a particle of mass `m` starting at rest is subject to a constant force `F` and simulated for time `t`
- **THEN** its displacement matches `F * t**2 / (2 * m)`

#### Scenario: Energy is conserved
- **WHEN** two particles joined by a spring are simulated with no drag
- **THEN** their total energy stays constant within solver tolerance

### Requirement: Ecology models
The library SHALL provide populations with exponential growth, logistic growth, mortality, and predation.

#### Scenario: Logistic saturation
- **WHEN** a population with logistic growth is simulated for a long time
- **THEN** its count approaches the carrying capacity

### Requirement: Epidemiology models
The library SHALL provide compartments with infection and recovery interactions, each with a deterministic mode and a
stochastic mode.

#### Scenario: Switch to stochastic
- **WHEN** a user sets the infection and recovery interactions to stochastic mode
- **THEN** the system compiles to a jump process and the compartment counts stay non-negative integers

### Requirement: Economics models
The library SHALL provide a Solow growth model in continuous time and a cobweb market model in discrete time.

#### Scenario: Solow steady state
- **WHEN** a Solow economy is simulated for a long time
- **THEN** its capital approaches `(s / delta) ** (1 / (1 - alpha))`

#### Scenario: Cobweb convergence
- **WHEN** a cobweb market with a supply slope smaller than the demand slope is simulated
- **THEN** its price converges to the equilibrium price

### Requirement: Finance models
The library SHALL provide an asset with geometric Brownian motion.

#### Scenario: Asset price stays positive
- **WHEN** an asset with geometric Brownian motion is simulated
- **THEN** its price stays positive and its trajectory is reproducible with a fixed random state

### Requirement: Reflecting box and ideal gas
The physics module SHALL provide a box with elastically reflecting walls for particles and particle groups, and a
helper that builds an ideal gas of a given size and temperature.

#### Scenario: Ideal gas law
- **WHEN** an ideal gas of N molecules at temperature kT is simulated in a square box of area A
- **THEN** the pressure measured from wall impulses times A equals N times kT within 2 percent

#### Scenario: Molecules stay in the box
- **WHEN** an ideal gas is simulated for any duration
- **THEN** every molecule stays inside the box and keeps its speed

### Requirement: Pair potentials for particle groups
The physics module SHALL provide Lennard-Jones and soft-sphere pair interactions with a cutoff, which run on the native
solvers and on the LAMMPS backend.

#### Scenario: Lennard-Jones minimum
- **WHEN** two members interact through a Lennard-Jones potential with `sigma = 1`
- **THEN** the force between them is zero at the distance `2 ** (1 / 6)`

### Requirement: Langevin thermostat
The physics module SHALL provide a Langevin thermostat that adds drag and random forces to a particle group, which runs
on the native SDE solvers and on the LAMMPS backend.

#### Scenario: Equilibrium temperature
- **WHEN** an ideal gas starts at temperature 0.5 under a Langevin thermostat whose `temperature` is 2
- **THEN** its kinetic temperature approaches 2 within 5 percent after many relaxation times

### Requirement: Social agent-based models
The library SHALL provide a `social` module with Schelling segregation, Boltzmann wealth exchange, and the voter model.

#### Scenario: Schelling segregation increases
- **WHEN** a Schelling model with a similarity threshold of 0.5 runs until no agent moves or 100 steps pass
- **THEN** the share of similar neighbours is higher at the end than at the start

#### Scenario: Wealth stays constant
- **WHEN** a Boltzmann wealth model runs for any number of steps
- **THEN** the total wealth of all agents stays equal to its initial value

#### Scenario: Voter consensus on a small network
- **WHEN** a voter model runs on a connected network of 20 agents for long enough
- **THEN** every agent ends with the same opinion

### Requirement: Agent-based predator and prey
The ecology module SHALL provide a wolf and sheep model on a grid with grass regrowth, in which agents gain energy by
eating, reproduce, and die.

#### Scenario: Populations change
- **WHEN** the wolf and sheep model runs for 100 steps with its default parameters
- **THEN** both the wolf and sheep counts change over the run and are reported every frame
