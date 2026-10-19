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

### Requirement: Coupled harvesting example
The library SHALL provide a socio-ecological model in which farmer agents harvest a resource whose stock regrows by a
logistic ODE, and the harvest depends on the number of farmers through an aggregate.

#### Scenario: Overharvesting collapses the resource
- **WHEN** the harvesting model runs with a harvest effort high enough that harvest exceeds the largest regrowth
- **THEN** the resource stock falls towards zero over the run

#### Scenario: Sustainable effort
- **WHEN** the harvesting model runs with a harvest effort below the largest regrowth
- **THEN** the resource stock settles at a positive level

### Requirement: Ready-made models are registered factories
Every ready-made model SHALL be a factory function registered in a catalog with its field, description, tags,
references, and default run settings, and the catalog SHALL list, describe, create, and run models by name.

#### Scenario: Browse the catalog
- **WHEN** a user lists the catalog filtered by the field `ecology`
- **THEN** the result holds every ecology model with its description, its references, and the formalism inferred from
  its laws

#### Scenario: Create by name
- **WHEN** a user creates `lotka_volterra` with a predation rate of 0.2
- **THEN** the result is a system that the simulation runs like the same model built from its components

### Requirement: Every registered model runs through the generic interface
Every registered model SHALL be created with its defaults, compiled, simulated, logged to Rerun, and cloned through the
same public interface, without code specific to the model.

#### Scenario: Conformance
- **WHEN** the conformance suite runs over the catalog
- **THEN** every model completes all five steps, and its trajectory holds every declared state and observable

### Requirement: The library uses only the public interface
Library modules SHALL build their models only from names that the core packages export.

#### Scenario: Private import
- **WHEN** a library module imports a private name of another package
- **THEN** the library import check fails and names the module and the import

### Requirement: Higher-order models across fields
The library SHALL provide, as registered factories written in their natural order:
- the nonlinear pendulum as Newton's law;
- a driven damped oscillator;
- a series RLC circuit as a second-order charge equation;
- Sprott's jerk equation;
- Samuelson's multiplier-accelerator;
- a Fibonacci-type recurrence.

#### Scenario: Circuit and oscillator agree
- **WHEN** the RLC circuit and the damped oscillator run with matching coefficients `L = m`, `R = c`, and `1 / C = k`
- **THEN** the charge and the displacement trajectories are identical

### Requirement: Nonlinear dynamics models
The library SHALL provide the Lorenz, Rössler, Hénon, logistic map, Duffing, Van der Pol, and Hénon-Heiles models as
registered factories with their standard parameters.

#### Scenario: Strange attractor
- **WHEN** a user runs the Lorenz model with its defaults for 50 time units
- **THEN** the trajectory stays bounded and visits both lobes of the attractor

### Requirement: Delay, event, and forced models
The library SHALL provide Mackey-Glass, Hutchinson's delayed logistic, delayed Kuramoto, a bouncing ball, Newton's
cradle, a circular billiard, integrate-and-fire and Izhikevich neurons, the forced Duffing oscillator, stochastic
resonance, and a Brownian particle near a wall with multiplicative noise as registered factories.

#### Scenario: Izhikevich regular spiking
- **WHEN** the Izhikevich neuron runs with the regular-spiking parameters and a constant input of 10
- **THEN** it fires periodically with each spike reset recorded as an event

### Requirement: Lattice models
The library SHALL provide elementary automata, Game of Life, majority rule, Langton's ant, Nagel-Schreckenberg traffic,
the hodgepodge machine, the Bak-Tang-Wiesenfeld sandpile, the Drossel-Schwabl forest fire, Olami-Feder-Christensen
earthquakes, and the Ising, Potts, and XY models as registered factories.

#### Scenario: Traffic jams
- **WHEN** the Nagel-Schreckenberg model runs at densities 0.1 and 0.5 with slowdown probability 0.3
- **THEN** the mean flow at density 0.5 is lower than at the critical density near 0.15, and jams appear as backward
  moving clusters of stopped cars

### Requirement: Field models
The library SHALL provide heat diffusion, Fisher-KPP, Gray-Scott, Brusselator, FitzHugh-Nagumo spirals, Schnakenberg
Turing patterns, and pheromone foraging as registered factories.

#### Scenario: Fisher front speed
- **WHEN** the Fisher-KPP model runs with growth rate `r` and diffusion `D` from a step profile
- **THEN** the front moves at a speed within 5% of `2 * sqrt(r * D)`

#### Scenario: Turing spots
- **WHEN** the Schnakenberg model runs in its Turing-unstable regime from small noise
- **THEN** a stationary pattern with a dominant wavelength within 10% of the linear prediction forms

### Requirement: PDE models
The library SHALL provide Kuramoto-Sivashinsky, complex Ginzburg-Landau, Swift-Hohenberg, Burgers' shock, the
shallow-water dam break, decaying two-dimensional turbulence, Rayleigh-Bénard convection, and heat on an L-shaped
domain as registered factories.

#### Scenario: Swift-Hohenberg rolls
- **WHEN** the Swift-Hohenberg model runs above onset from small noise
- **THEN** a roll pattern forms with a wavelength within 5% of 2π
