# Spec Delta

## MODIFIED Requirements

### Requirement: Basin stability is an experiment
The experiments layer SHALL estimate the basin stability of the attractors of a system by running it from initial
states sampled within bounds and classifying each final state, as the experiment function `basin_stability` whose
result holds the fraction of runs per attractor. With `vectorized=True` it SHALL run the initial states as batches of
runs of one compiled problem, for systems with groups and child systems as for systems of single elements.

#### Scenario: Bistable system
- **WHEN** a user runs a basin stability experiment of a symmetric double-well system over symmetric bounds of the
  initial position
- **THEN** each of the two attractors gets a fraction within 0.05 of one half

#### Scenario: System of groups
- **WHEN** a user runs basin stability of a Kuramoto system of 100 oscillators over 1000 initial states with
  `vectorized=True`
- **THEN** it returns the same fractions as with `vectorized=False`

## ADDED Requirements

### Requirement: Experiments of runs run in batches
`ensemble`, `sweep`, `compare_scenarios`, `basin_stability`, and `orbit_diagram` SHALL take `vectorized` and
`batch_size`, and with `vectorized=True` SHALL run the runs that share a compiled structure as batches, one batch per
task of the executor.

#### Scenario: Vectorised sweep
- **WHEN** a user sweeps the growth rate of the logistic map over 200 values with `vectorized=True`
- **THEN** the result equals the result with `vectorized=False`

#### Scenario: Reaction networks batch
- **WHEN** an ensemble of 50 runs of a stochastic SIR reaction network executes with `vectorized=True`
- **THEN** it returns 50 runs and raises no role error

#### Scenario: Scenarios with schedules batch
- **WHEN** two scenarios, one of which halves a parameter at time 50, run 20 replicates each with `vectorized=True`
- **THEN** the result equals the result with `vectorized=False`

### Requirement: Batched experiments are benchmarked against runs one by one
The benchmarks SHALL time batched experiments against the current runs one task per run on the same machine.

#### Scenario: Ensemble of a stochastic reaction network
- **WHEN** 100000 replicates of a stochastic SIR reaction network of 1000 people run with `GillespieSolver` by
  `ensemble`, with `vectorized=True` on the Numba backend and with `executor='process'`, in a test marked `benchmark`
- **THEN** the batched run takes at most the time of the process run

#### Scenario: Orbit diagram on a GPU
- **WHEN** an orbit diagram of the logistic map over 1000000 values of `r` runs with `vectorized=True` on
  `JaxBackend(device='gpu')`, in a test marked `benchmark` and `gpu`
- **THEN** it takes at most a fifth of the time of the same diagram on the Numba backend with `vectorized=True`
