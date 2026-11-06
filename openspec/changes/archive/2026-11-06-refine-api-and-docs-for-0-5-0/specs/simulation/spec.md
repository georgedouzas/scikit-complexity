# Spec Delta

## ADDED Requirements

### Requirement: A simulation holds a default end time
A simulation SHALL take an optional end time `until` as an argument. `run()` without an end time SHALL run to it, an
end time passed to `run` SHALL take precedence, and a run with neither SHALL raise a validation error that asks for one.

#### Scenario: Stored end time
- **WHEN** a user calls `Simulation(system, until=20).prepare().run()`
- **THEN** the run ends at time 20

#### Scenario: Passed end time wins
- **WHEN** a user calls `Simulation(system, until=20).prepare().run(until=5)`
- **THEN** the run ends at time 5

#### Scenario: No end time
- **WHEN** a user calls `Simulation(system).prepare().run()`
- **THEN** it raises a validation error that shows how to pass an end time

## MODIFIED Requirements

### Requirement: Simulations take observables
A simulation SHALL take the observables to record as an argument. A catalog entry SHALL declare the standard observables
of its system, which `create_simulation` passes to the simulation. Trajectories SHALL record observables as before.

#### Scenario: Observables given to a simulation
- **WHEN** a user simulates a system with `Simulation(system, observables=[total_population])`
- **THEN** the trajectory records `total_population` at every frame

#### Scenario: Catalog observables
- **WHEN** a user runs a catalog system with `create_simulation(name).prepare().run()`
- **THEN** the trajectory records the observables declared by its catalog entry

### Requirement: A model package plugs in its own compiler
A compiler defined outside the engine, on `BaseCompiler`, SHALL compile a system passed to `Simulation` with
`compiler=`, and the simulation SHALL run the solver that compiler names, without the engine
importing the package.

#### Scenario: Rewriting from the models library
- **WHEN** `Simulation(create_system('koch_curve'), compiler=RewritingCompiler())` runs, with `RewritingCompiler` imported
  from `skcomplex.core.engine.compile`
- **THEN** the run rewrites the word once per frame, and `skcomplex.core.engine` imports nothing from `skcomplex.systems`

#### Scenario: Quantum system without its compiler
- **WHEN** a quantum system runs with the default compiler
- **THEN** checking it raises a validation error that names `compiler=QuantumCompiler()`
