# search-processes Specification

## Purpose
Lets a user build optimisers as processes: a population of candidate solutions moving on an objective landscape, one
generation per time step.

## Requirements

### Requirement: Optimisation moves a population of candidates
`OptimizationInteraction` SHALL bind a group of candidates holding a vector state in a box, an objective of the array of
candidates, and an algorithm spec, `DifferentialEvolutionSpec`, `ParticleSwarmSpec`, `EvolutionStrategySpec`, or
`CovarianceAdaptationSpec`. It SHALL advance the population by one generation per step of discrete time, write the
candidates into the members, and report the `best` and `mean` objective values every step. Runs with the same seed
SHALL be identical.

#### Scenario: Differential evolution on Rastrigin
- **WHEN** `OptimizationInteraction` with `DifferentialEvolutionSpec()` moves 40 candidates on the two-dimensional
  Rastrigin function in `[-5.12, 5.12]` with `random_state=1`
- **THEN** the reported best value is below 1e-3 by step 50 and never increases from one step to the next

#### Scenario: Algorithm that is not a spec
- **WHEN** an `OptimizationInteraction` takes the string `'ga'` as its algorithm
- **THEN** checking it raises a `ValidationError` that lists the algorithm specs

### Requirement: Annealing follows a cooling temperature
`SamplingInteraction` SHALL take `cooling`, a function of time that multiplies its temperature, so a Metropolis
acceptance with the `'gaussian'` move anneals a group of candidates whose energy reads no neighbours.

#### Scenario: Cooling
- **WHEN** simulated annealing runs with `cooling` falling geometrically to 1e-3 of the initial temperature
- **THEN** the share of proposals accepted falls with the temperature, from above 0.8 at the start to below 0.2 in the
  last tenth of the run, and the energy of a quadratic objective ends within 0.01 of its minimum
