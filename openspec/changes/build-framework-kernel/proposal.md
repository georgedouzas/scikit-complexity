# Proposal

## Why

The current package models only point particles with pairwise forces in Euclidean space, depends on an undeclared Sage
install, and has a broken `simulate` method. The project's goal is a framework that can describe models from any
science and run them live, so the kernel has to be rebuilt around a domain-agnostic description, a symbolic layer
based on SymPy, real-time execution, and Rerun visualization.

## What Changes

- **BREAKING** Replace `skcomplex.base`, `skcomplex.component`, and `skcomplex.system` with a layered package of
  `core`, `model`, `compile`, `solvers`, `runtime`, `viz`, `analysis`, and `models`.
- Add a declarative description of models: elements with typed state and parameters, interactions with roles and
  laws, spaces, and nested systems that compose.
- Add law kinds `Rate`, `Noise`, `Jump`, and `Update`, each accepting a SymPy expression or a Python callable, and
  infer the mathematical form of a system (ODE, SDE, jump process, discrete map) from the law kinds it contains.
- Add solvers for each of the four forms, a `Simulation` runtime with real-time pacing, live parameter changes, a
  background mode, observers, and a recorded `Trajectory`.
- Add Rerun visualization as a core feature, with entity paths derived from the system tree and visuals declared by
  element types.
- Add symbolic analysis that returns the equations of a system and solves them exactly where SymPy can, and accept
  Sage expressions through the optional `sage` extra.
- Add plug-and-play model modules for physics, ecology, epidemiology, economics, and finance.
- **BREAKING** Make `numpy`, `scipy`, `sympy`, `rerun-sdk`, and `scikit-learn` runtime dependencies, move `lammps`
  to the `lammps` extra, add the `sage` extra, and support Python 3.11 to 3.13.
- Fill the constitution's Project Profile with the ecosystem contract and the dependency decisions.

## Capabilities

### New Capabilities

- `system-modeling`: describing elements, interactions, laws, spaces, and nested systems.
- `model-compilation`: turning a system into a problem of one mathematical form with numeric functions.
- `simulation`: solving problems, pacing runs in real time, changing parameters live, and recording trajectories.
- `rerun-visualization`: streaming a running simulation to the Rerun viewer or a recording file.
- `symbolic-analysis`: deriving the equations of a system, solving them exactly, and accepting Sage expressions.
- `model-library`: ready-made models for physics, ecology, epidemiology, economics, and finance.

### Modified Capabilities

None. The project has no existing specs.

## Impact

- All public imports change. `ClassicalMechanicsSystem`, `PointParticle`, `EuclideanSpace`, and the force and
  potential classes are replaced by `System`, `skcomplex.models.physics.Particle`, `EuclideanSpace`, and physics
  interactions.
- Sage is no longer required. Tests that used Sage symbols move to SymPy, and one test covers the Sage extra.
- `pyproject.toml`, `pdm.lock`, the README usage section, and the documentation change.
