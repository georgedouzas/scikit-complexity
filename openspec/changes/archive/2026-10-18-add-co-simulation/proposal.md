# Proposal

## Why

Many complex systems join parts that need different mathematics: farmers modelled as agents act on soil and water
modelled as differential equations, a stochastic epidemic in a town drives an economy described by a discrete map, and
particles in LAMMPS heat a fluid described by a PDE. Today a system must have one form, so mixing law kinds raises an
error and such models cannot be built.

## What Changes

- Split a system whose law kinds do not fit one form into partitions, one per subsystem whose laws do fit one form, and
  give each partition its own problem and solver.
- Let a law read states of other partitions as inputs, held constant between synchronisation points.
- Add aggregates, which turn an agent group or element group into scalar values such as a count, sum, or mean that
  other partitions read like states.
- Let agent rules read the states of other partitions by dotted name.
- Add a co-simulation scheduler that advances all partitions to each synchronisation point with a Jacobi or
  Gauss-Seidel scheme and a configurable synchronisation interval.
- Keep one runtime: frames, trajectories, live parameters, clocks, and Rerun visualization cover all partitions with
  the same dotted names.
- Add a socio-ecological example to the model library: agent farmers harvesting a resource that regrows by an ODE.

## Capabilities

### New Capabilities

- `co-simulation`: partitioning systems by form, inputs across partitions, aggregates, and the scheduler.

### Modified Capabilities

- `model-compilation`: form inference applies per partition instead of rejecting every mix of law kinds.
- `model-library`: a coupled agent and ODE example.

## Impact

- `skcomplex.compile` returns a partitioned problem when a system has several forms. `skcomplex.runtime` gains the
  scheduler, which `Simulation` uses for partitioned problems.
- Systems with one form compile and run exactly as before.
- Depends on `add-element-groups`, `add-lammps-backend` for the backend contract, and `add-mesa-agents` for agent
  partitions and aggregates over agents.
