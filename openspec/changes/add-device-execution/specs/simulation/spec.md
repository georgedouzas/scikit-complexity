# Spec Delta

## ADDED Requirements

### Requirement: Seeded JAX runs repeat on one device
A stochastic run on the JAX backend SHALL draw from a JAX key derived from the `random_state` of `system.simulate`, an
integer or a NumPy generator, or from the `SeedSequence` an experiment spawns for the run. Two runs with the same random
state on the same device SHALL give the same trajectory within 1e-10, and the runs of an experiment SHALL draw from
distinct keys.

#### Scenario: Same seed on the CPU device
- **WHEN** `SIRSystem` as a stochastic reaction network with `solver=GillespieSolver()` and
  `backend=JaxBackend(device='cpu')` is prepared once and simulated twice with `random_state=3`
- **THEN** both trajectories are equal

#### Scenario: Spawned sequences
- **WHEN** an ensemble of two runs of the same system with `random_state=7` executes on the JAX backend
- **THEN** the two runs differ
