# Design

## Context

Groups compile one law over many members, rate laws give ODEs, and co-simulation aggregates groups into values other
partitions read. Bellomo et al. (2017 §3.4.1) show that a discrete activity grid turns the kinetic equations into
ODEs, which every existing ODE solver and backend handles.

## Goals / Non-Goals

**Goals:**

- Write a kinetic model as its encounter structure, as Bellomo et al. (2017 ch. 3) do, and get the gain-loss ODEs
  without deriving them by hand.
- Move between agent, distribution, and aggregate levels of the same model.

**Non-Goals:**

- Continuous activity with quadrature-based collision operators. A fine grid approximates it.
- Spatial kinetic models with velocity and position. Crowds use the network version.

## Decisions

### Distribution states

`Distribution(label, activities, subsystems=('f',), initial=...)` creates one state per activity value and subsystem.
Each state is the fraction or density at that activity, stored as a group whose members are activity values, so
kernels stay vectorised. On a network, the distribution has one copy per node.

### Encounters

`Encounter(candidate, field, rate, transition, test=None)` declares that a candidate particle at activity `h` meets a
field particle at `k` at rate `η(h, k)` and moves to `i` with probability `A(h, k; i)`. Probabilities must sum to one
over `i`, which the compiler checks symbolically. The derived rate for subsystem `f` at activity `i` is the gain
`Σ_hk η A f_h f_k` minus the loss `f_i Σ_k η f_k`. `Proliferation` and `Destruction` add `μ(h, k) f_h f_k` terms
without conservation, and `Relaxation(target, equilibrium, rate)` adds `rate (f_eq - f)`. Transition tables, given as
symbolic functions or arrays, are lowered to tensor contractions on NumPy and fused loops on Numba.

### External actions

`ExternalAction(target, kind='micro' | 'macro', ...)` either adds an extra encounter with a fixed field distribution,
such as a policy acting on wealth classes, or a term that depends on moments, such as a control proportional to the
mean.

### Particle solver

`ParticleKinetics(n)` samples `n` particles per subsystem. Each step it draws encounter pairs with probabilities from
the rates and applies transitions, proliferation, and destruction to the sampled particles. It returns the empirical
distributions on the grid, so trajectories match the ODE form. Its convergence with `n` is an acceptance test.

### Level conversions

`Distribution.from_agents(group, variable, activities)` bins an agent group into a distribution at a frame, and
`agents_from(distribution, agent_type, n)` samples agents from one. Co-simulation can then run one level in a
partition and feed another level its moments.

## Risks / Trade-offs

- [Cost grows with the cube of the grid size] → Transition tables are sparse in the references' models, and the
  lowering skips zero entries.
- [Probabilities that do not sum to one] → The compiler rejects them with the activity and the sum.
