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

## Implementation Notes

- `Field(label, element, space, boundary=..., values=...)` is a group with a lattice or network space, so a field's
  states and parameters use the group layout with one row per site, and its laws are the element type's laws for one
  site. Lattice-shaped values are flattened when the compiler reads them, so fields clone like any estimator.
- Operators apply to any expression of a field's states and parameters, not only to states. The compiler lowers each
  operator by evaluating its expression at neighbouring sites, through symbols with a new neighbour source that reads
  the value of every member's neighbour. Nested operators compose their offsets, so `divergence(gradient(u))` and
  `laplacian(laplacian(u))` work on lattices. Networks take one step, the graph Laplacian `sum_j (u_j - u_i)`.
- Boundary conditions choose the neighbour table: periodic edges wrap, zero-flux edges mirror the lattice so the
  ghost equals the edge value and the total is conserved to rounding, and fixed edges read a ghost value. Parameters
  read across edges mirror unless the lattice is periodic.
- On Numba, neighbour reads compute the neighbour's index from the member index and the lattice shape inside the
  compiled kernel, and the shape and offset are written into the generated source, so its cache key distinguishes
  lattices. Network stencils keep the NumPy kernels, which the backend's per-kind fallback already allowed.
- The equations replace the stencils by the operators as written, for `equations_of` and the analysis toolkit.
- The stable step is `h^2 / (degree D)` with the degree `2d` on lattices and the largest degree on networks, from
  Laplacian coefficients that are numbers at the initial values. Every fixed-step stepper whose `dt` exceeds it warns
  when it is initialized.
- Agents couple through `model.field` and `model.deposit` on the model view. Deposits collect in the agent stepper and
  the co-simulation adds them to the state at every synchronisation point. Agent cells are lattice coordinates, so a
  grid of width `w` and height `h` matches a lattice of shape `(w, h)`.
- The FitzHugh-Nagumo spiral needs the slow recovery `epsilon = 0.02`. With `0.08`, the broken wave leaves the
  default domain before it curls.
- The pheromone foraging ants have a mild outward bias, as a nest scent would give, because an unbiased random walk
  rarely finds the food in a few hundred steps. The trail is about four times the background after 600 steps.
- The Turing scenario runs a line of 256 sites, where the dominant wavelength 14.2 is within 5% of the predicted 14.9
  and fixed from the first frames, with the remaining change still shrinking as the pattern settles.
