# Design

## Context

Groups compile one symbolic law for all members and evaluate it in member mode, and lattice spaces from
`add-lattice-dynamics` define sites and neighbourhoods. A field is a group with one member per site whose laws can read
neighbouring members through operators.

## Goals / Non-Goals

**Goals:**

- Reaction-diffusion models written as their equations, for example `Rate(u, D * laplacian(u) - u * v**2 + F * (1 -
  u))`.
- Fields on networks with the graph Laplacian, for diffusion on networks (Thurner et al. 2018 §4.8).

**Non-Goals:**

- Spectral methods, finite volumes, finite elements, implicit and exponential steppers for stiff fields, and
  Navier-Stokes turbulence (Satz 2022 ch. 12, Mihailović et al. 2023 ch. 8). `add-pde-solvers` adds them on top of
  the fields of this change.

## Decisions

### Fields

`Field(label, space, **states)` creates an element group with one member per site of a lattice or node of a network.
Its states are fields, and its laws are written for one site as for any group. `space` must be a `LatticeSpace` with
spacing or a `NetworkSpace`.

### Operators

`laplacian(u)`, `gradient(u)`, `divergence(vector)`, and `advection(u, velocity)` are Sage functions that stay symbolic
in equations. The lowering replaces them with stencil expressions:
- the five-point (or 2d+1-point) Laplacian divided by the squared spacing, as Datseris and Parlitz (2022 ch. 11) use;
- central differences for gradients;
- upwind differences for advection;
- `L u` with the graph Laplacian on networks.

The Numba backend fuses the stencil, and the symbolic equations keep the operators for display.

### Boundary conditions

`Field(..., boundary={'u': 'periodic' | 'zero-flux' | ('fixed', value)})` sets ghost values per field, so the same law
holds on edge sites.

### Stability

For explicit steppers, the compiler computes the diffusive limit `dt ≤ h² / (2 d D_max)` from the Laplacian
coefficients when they are numeric, and warns when the frame step exceeds it, naming the suggested value.

### Coupling with agents

Agents on a lattice read the field at their cell through `model.field(label, cell)` and write increments through
`model.deposit(label, cell, amount)`. These are applied as held inputs at synchronisation points, which keeps the field
and agent partitions separate under co-simulation.

## Risks / Trade-offs

- [Stiffness on fine grids] → The stability warning and the BDF solver cover the library, and `add-pde-solvers` adds
  implicit and exponential steppers.
- [Field memory in trajectories] → Fields are thinned with the existing logging and recording options.
