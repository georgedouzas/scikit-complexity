# Spec Delta

## MODIFIED Requirements

### Requirement: A model package plugs in its own compiler
A compiler defined outside the engine, on `BaseCompiler`, SHALL compile a system passed to `Simulation` or
`compile_system` with `compiler=`, and the simulation SHALL run the solver that compiler names, without the engine
importing the package.

#### Scenario: Rewriting from the models library
- **WHEN** `Simulation(create('koch_curve'), compiler=RewritingCompiler())` runs, with `RewritingCompiler` imported
  from `skcomplex.engine.compile`
- **THEN** the run rewrites the word once per frame, and `skcomplex.engine` imports nothing from `skcomplex.models`

#### Scenario: Quantum system without its compiler
- **WHEN** a quantum system runs with the default compiler
- **THEN** checking it raises a validation error that names `compiler=QuantumCompiler()`
