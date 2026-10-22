# Design

## Context

Groups compile one law over many members, rate laws give ODEs, and co-simulation aggregates groups into values other
partitions read. Bellomo et al. (2017 §3.4.1) show that a discrete activity grid turns the kinetic equations into
ODEs, which every existing ODE solver and backend handles.

After `redesign-system-model`, a system is its components, interactions, space, and time. A distribution over a
discrete activity grid is a group on the sites of a lattice over that grid: one member per activity value, one state
per functional subsystem. Observables belong to simulations and catalog entries, and every law, solver, and backend
declares what it requires and supports.

## Goals / Non-Goals

**Goals:**

- Write a kinetic model as its encounter structure, as Bellomo et al. (2017 ch. 3) do, and get the gain-loss ODEs
  without deriving them by hand.
- Move between agent, distribution, and aggregate levels of the same model.

**Non-Goals:**

- Continuous activity with quadrature-based collision operators. A fine grid approximates it.
- Spatial kinetic models with velocity and position. Crowds use the network version.
- A distribution component. A distribution is a site group on an activity lattice.

## Decisions

### Distributions are site groups on an activity lattice

A distribution over `n` activity values is `Group(label, element=E, size=n, on='sites')` in a system whose space is
`LatticeSpace(n, boundary='open')`. The element `E` declares one real state per functional subsystem, for example
`f`, and each member holds the fraction or density at its activity value, so kernels stay vectorised. The activity
values are a constant array given to the encounter laws and the moments. No kinetic-specific component is added.

### Encounters

`Encounter(candidate, field, rate, transition, test=None)` in `core.laws` declares that a candidate particle at
activity `h` meets a field particle at `k` at rate `η(h, k)` and moves to `i` with probability `A(h, k; i)`. It is
returned by an interaction whose roles bind the candidate, field, and test groups. The derived rate for subsystem `f`
at activity `i` is the gain `Σ_hk η A f_h f_k` minus the loss `f_i Σ_k η f_k`. `Proliferation` and `Destruction` add
`μ(h, k) f_h f_k` terms without conservation. Transition tables, given as symbolic functions or arrays, are lowered to
tensor contractions on NumPy and fused loops on Numba.

Relaxation toward an equilibrium is a `Rate` law `rate * (f_eq - f)` and needs no primitive of its own.

The three laws share the law kind `encounter` of the continuous class, and declare:

```text
Requires(time={continuous}, space={lattice}, domains={real})
```

Validation reports:
- an encounter in a system without continuous time or without a lattice space, with the fix of declaring them;
- a transition probability that does not sum to one over the outcomes of an encounter, with the encounter's path, the
  activities, and the sum, which the validator checks symbolically or numerically before compilation.

The ODE solvers and the NumPy and Numba backends extend their `Supports.laws` with `encounter`, since the compiler
lowers encounters to rates.

### External actions

External actions need no primitive of their own:
- a micro-scale action, such as a policy acting on wealth classes, is an `Encounter` whose field role binds a group
  with fixed states;
- a macro-scale action, such as a control proportional to the mean, is a `Rate` law that reads moments.

### Moments

`moment(f, order, coordinates=activities)` is a reduction expression next to `total`, `mean`, and `variance`. It is the
same reduction `add-phase-space-fields` adds over velocity and orientation axes, with the activity values as
coordinates, and whichever change lands second reuses the first one's. The
density is the order-0 moment, the mean activity is the order-1 moment divided by the density, and the variance and
higher moments follow. Moments can be read inside laws and passed as observables to `Simulation(observables=...)` and
catalog entries.

### Particle solver

`ParticleKinetics(n)` in `core.solvers` samples `n` particles per subsystem. Each step it draws encounter pairs with
probabilities from the rates and applies transitions, proliferation, and destruction to the sampled particles. It
returns the empirical distributions on the grid, so trajectories match the ODE form. Its convergence with `n` is an
acceptance test. It declares:

```text
Supports(time={continuous}, space={lattice}, laws={encounter}, domains={real}, components={site-group})
```

With this solver, validation reports any law other than an encounter, such as a rate law, and lists the ODE solvers
that support it.

### Level conversions

`distribution_from(group, state, activities)` in `core.components` bins the members of a group into the initial
states of a distribution site group at a frame, and `members_from(distribution, element, n)` samples `n` members of a
variable group of `element` from one. Co-simulation can then run one level in a partition and feed another level its
moments.

### Library models

The kinetic models live in their field packages: `biology/oncology`, `economics/econophysics`,
`social/opinion_dynamics`, and `social/crowds`. Their transition tables and encounter rates are realizations and stay
in those packages.

## Risks / Trade-offs

- [Cost grows with the cube of the grid size] → Transition tables are sparse in the references' models, and the
  lowering skips zero entries.
- [Probabilities that do not sum to one] → Validation reports them with the activity and the sum before compilation.

## Open Questions

- Distributions on network nodes. The redesign holds a distribution as a site group on a lattice over the product
  space, but the core has no product of a network and an activity lattice. Should a distribution on a network be a
  site group on a two-dimensional lattice indexed by node and activity, with encounters reading the network's
  adjacency from a separate parameter; a nested system per node; or should a product space kind be added?

  Resolved for this change with the simplest option. `evacuation` holds its crowd as a site group on a
  two-dimensional lattice indexed by room and speed class, with one last row of sites for the outside, and its walking
  encounters build their tables from the network of rooms. The network is a balanced tree given by numeric
  parameters, since interaction parameters hold numbers only. A product space kind stays open for a later change.
