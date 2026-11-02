# Proposal

## Why

Pattern formation and transport need continuous fields over space, which the framework cannot express. Examples:
- **Reaction-diffusion:** Gray-Scott, Brusselator, Fisher-KPP fronts, Turing patterns, and FitzHugh-Nagumo spiral waves
  (Datseris and Parlitz 2022 ch. 11; Gros 2015 ch. 4; Downey's *Think Complexity* ch. 7; Trzaska 2023 ch. 4).
- **Information flows** in such patterns (Lindgren 2024 ch. 7).
- **Stigmergy:** pheromone fields that agents read and write (Argun et al. 2021 ch. 15; Carmichael et al. 2019).
- **Convection:** the heat and convection equations behind Satz (2022 ch. 12).

Ceccherini-Silberstein and Coornaert (2023 ex. 1.4.3) note that the discrete Laplacian is itself a linear cellular
automaton. Fields on a lattice or a network discretised by the method of lines therefore become ordinary rate laws that
every existing solver handles.

## What Changes

- Add fields: states with one value per site of a lattice or one value per node of a network, declared on an element
  that spans the space.
- Add differential operators on fields: Laplacian, gradient, divergence, and advection. They lower to finite-difference
  stencils on lattices and to graph Laplacians on networks.
- Add boundary conditions on fields: periodic, zero flux, and fixed value.
- Compile fields with rate laws to ODE systems by the method of lines, run by the existing solvers and backends. Add a
  stability check that suggests a step size for explicit steppers.
- Couple fields with agents and particles: agents read and write the field at their cell, and fields act as inputs in
  co-simulation.
- Log fields to Rerun as images with a color map.
- Add library models: heat diffusion, Fisher-KPP, Gray-Scott, Brusselator, FitzHugh-Nagumo spirals, Schnakenberg
  Turing patterns, and pheromone foraging.

## Capabilities

### New Capabilities

- `spatial-fields`: fields over lattices and networks, differential operators, boundary conditions, and coupling with
  agents.

### Modified Capabilities

- `model-library`: field models.

## Impact

- `skcomplex.model` gains `Field`, `laplacian`, `gradient`, `divergence`, and `advection`, and `skcomplex.compile`
  lowers them to stencils.
- Depends on `add-lattice-dynamics` for lattice spaces and on `add-general-model-interface` for observables.
