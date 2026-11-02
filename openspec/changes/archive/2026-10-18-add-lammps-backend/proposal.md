# Proposal

## Why

Mature libraries already solve whole classes of models faster and more completely than a general framework can.
LAMMPS integrates particle systems with hundreds of pair potentials, thermostats, and walls in optimized C++, and its
PyPI wheel ran 9612 Lennard-Jones atoms at 734 steps per second in a probe. The framework should describe a model once
and hand it to such a library, while keeping its runtime, trajectories, and Rerun visualization.

## What Changes

- Add external backends: solvers that translate a compiled problem into a third-party library, check that every law
  is supported before running, and exchange state with the runtime every frame.
- **BREAKING** Make `lammps` a runtime dependency, with `mpich` on platforms where the LAMMPS wheel links MPI without
  bundling it, and remove the `lammps` extra.
- Add a LAMMPS backend for particle groups in Euclidean space: symbolic pair energies become `pair_style lepton`
  expressions, boxes become LAMMPS boundaries and `fix wall/reflect`, drag and thermostats become `fix langevin`, constant
  forces become `fix addforce`, and other single-particle forces run through `fix external` with the compiled kernels.
- Add `LennardJones`, `SoftSphere`, and `LangevinThermostat` to the physics module, with native and LAMMPS
  implementations.

## Capabilities

### New Capabilities

- `external-backends`: handing a whole problem to a third-party solver library, with LAMMPS as the first backend.

### Modified Capabilities

- `model-library`: Lennard-Jones and soft-sphere pair interactions and a Langevin thermostat.

## Impact

- New `skcomplex.backends` subpackage below `runtime` and above `solvers`. `Simulation` accepts a backend instance as
  its solver.
- `pyproject.toml` gains `lammps` and `mpich` as dependencies and loses the `lammps` extra.
- Depends on `add-element-groups` for particle groups and on `add-numba-acceleration` for `fix external` kernels.
