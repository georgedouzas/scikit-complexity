# Spec Delta

## Purpose

Lets a user model populations of active particles as distributions over an activity variable, held as site groups on
an activity lattice and changed by encounters with conservative, proliferative, and destructive outcomes, and move
between agent, distribution, and aggregate levels.

## ADDED Requirements

### Requirement: Distributions are site groups over activities
A distribution SHALL be a group on the sites of a lattice over a discrete grid of activity values, with one state per
functional subsystem, without any distribution-specific component in the core.

#### Scenario: Normalised start
- **WHEN** a user creates a wealth distribution over 9 classes as a site group on a lattice of 9 sites from an initial
  profile
- **THEN** the states hold the profile, and the order-0 moment equals its sum

### Requirement: Encounters derive gain and loss equations
Encounter laws SHALL derive the gain and loss equations of the kinetic theory from encounter rates and transition
probabilities. Conservative encounters SHALL keep the density constant. Proliferative and destructive encounters SHALL
change it as declared. Encounter laws SHALL be of the law kind `encounter`, in the continuous class, and SHALL declare
that they require continuous time, a lattice space, and real states.

#### Scenario: Conservative encounters
- **WHEN** a wealth model with only conservative encounters runs for 100 time units
- **THEN** the total density stays constant within 1e-10 relative error

#### Scenario: Invalid transition
- **WHEN** a transition probability sums to 0.9 over the outcomes of one encounter
- **THEN** validation reports an issue whose path is the encounter and whose message names the activities and the sum,
  before compilation

#### Scenario: Wrong substrate
- **WHEN** an encounter law is placed in a system with discrete time
- **THEN** validation reports that encounters need continuous time and suggests declaring `ContinuousTime()`

### Requirement: External actions act on transitions or moments
Systems SHALL express external actions at the micro scale through encounters whose field role binds a group with fixed
states, and at the macro scale through rate laws of the distribution's moments.

#### Scenario: Welfare policy
- **WHEN** a welfare policy transfers activity from the richest to the poorest classes
- **THEN** the Gini coefficient of the stationary distribution is lower than without the policy

### Requirement: Moments are reductions
The core SHALL provide the density, mean, variance, and higher moments of a distribution as reduction expressions,
readable inside laws and usable as observables of simulations and catalog entries.

#### Scenario: Mean activity
- **WHEN** a simulation records the mean activity moment of a distribution
- **THEN** it returns the activity-weighted sum divided by the density at every frame

### Requirement: A particle solver samples the same model
The same declarations SHALL run with a stochastic particle solver that converges to the ODE solution as the number of
particles grows. The solver SHALL declare that it supports continuous time, lattice spaces, site groups, and encounter
laws only.

#### Scenario: Convergence with particles
- **WHEN** a model runs with the particle solver at 10³, 10⁴, and 10⁵ particles over ensembles
- **THEN** the mean error of the distribution against the ODE solution falls roughly as one over the square root of the
  particle number

#### Scenario: Unsupported law with the particle solver
- **WHEN** a system holding a rate law is validated with the particle solver
- **THEN** validation reports that the solver does not support rate laws and lists the solvers that do

### Requirement: Levels convert into each other
The core SHALL build distributions from the states of a group's members and sample the members of a variable group from
distributions.

#### Scenario: Agents to distribution
- **WHEN** a user bins 1000 agents' wealth into a distribution over 9 classes
- **THEN** the distribution's density is 1 and its mean equals the agents' mean wealth class
