# Spec Delta

## ADDED Requirements

### Requirement: Seeded JAX runs repeat on one device
A stochastic run on the JAX backend SHALL draw from a JAX key derived from its random state, so two runs with the same
random state on the same device SHALL give the same trajectory within 1e-10, and runs from the spawned sequences of an
experiment SHALL draw from distinct keys.

#### Scenario: Same seed on the CPU device
- **WHEN** a stochastic SIR reaction network runs twice with `GillespieSolver`, `JaxBackend(device='cpu')`, and random
  state 3
- **THEN** both trajectories are equal

#### Scenario: Spawned sequences
- **WHEN** two runs take the first two children of `SeedSequence(7).spawn(2)` as their random states on the JAX backend
- **THEN** their trajectories differ
