# Spec Delta

## Purpose

Lets a user model populations of active particles as distributions over an activity variable, changed by encounters
with conservative, proliferative, and destructive outcomes, and move between agent, distribution, and aggregate levels.

## ADDED Requirements

### Requirement: Distributions are states over activities
An element SHALL hold a distribution per functional subsystem over a discrete grid of activity values, optionally on
every node of a network.

#### Scenario: Normalised start
- **WHEN** a user creates a wealth distribution over 9 classes from an initial profile
- **THEN** the states hold the profile, and the density observable equals its sum

### Requirement: Encounters derive gain and loss equations
Encounter laws SHALL derive the gain and loss equations of the kinetic theory from encounter rates and transition
probabilities. Conservative encounters SHALL keep the density constant. Proliferative and destructive encounters SHALL
change it as declared.

#### Scenario: Conservative encounters
- **WHEN** a wealth model with only conservative encounters runs for 100 time units
- **THEN** the total density stays constant within 1e-10 relative error

#### Scenario: Invalid transition
- **WHEN** a transition probability sums to 0.9 over the outcomes of one encounter
- **THEN** compiling raises a named error that names the encounter, the activities, and the sum

### Requirement: External actions act on transitions or moments
Systems SHALL apply external actions at the micro scale through extra encounters, and at the macro scale through terms
of the distribution's moments.

#### Scenario: Welfare policy
- **WHEN** a welfare policy transfers activity from the richest to the poorest classes
- **THEN** the Gini coefficient of the stationary distribution is lower than without the policy

### Requirement: Moments are observables
The modelling layer SHALL provide the density, mean, variance, and higher moments of distributions as observables.

#### Scenario: Mean activity
- **WHEN** a trajectory of a distribution is asked for the observable `mean`
- **THEN** it returns the activity-weighted mean divided by the density at every frame

### Requirement: A particle solver samples the same model
The same declarations SHALL run with a stochastic particle solver that converges to the ODE solution as the number of
particles grows.

#### Scenario: Convergence with particles
- **WHEN** a model runs with the particle solver at 10³, 10⁴, and 10⁵ particles over ensembles
- **THEN** the mean error of the distribution against the ODE solution falls roughly as one over the square root of the
  particle number

### Requirement: Levels convert into each other
The modelling layer SHALL build distributions from agent groups and sample agent groups from distributions.

#### Scenario: Agents to distribution
- **WHEN** a user bins 1000 agents' wealth into a distribution over 9 classes
- **THEN** the distribution's density is 1 and its mean equals the agents' mean wealth class
