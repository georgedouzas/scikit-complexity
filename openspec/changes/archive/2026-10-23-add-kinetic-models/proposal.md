# Proposal

## Why

Bellomo et al. (2017) model living systems with a kinetic theory of active particles. Each functional subsystem is a
distribution `f(t, u)` over an activity variable `u`, such as wealth, aggressiveness, or tumour progression. Encounters
at given rates change the distribution through transition probabilities that conserve, create, or destroy particles,
and the macroscopic quantities are the distribution's moments (Bellomo et al. 2017 §3.2, Eq. 3.17). The same
framework covers:
- criminality against security forces (§5.4.3);
- tumour-immune competition (§5.5.2);
- welfare and wealth distribution;
- crowds and traffic (ch. 6).

It links the agent level and the aggregate level through a middle, distributional level (§7.3), which Mobus (2022 §8.4)
also calls for when generating models at a chosen resolution. The framework has agents and aggregates, but no encounter
law between distributions, and no way to move between the agent and distribution levels.

**Rebased on `redesign-system-model`.** The redesign already provides the distribution state: a distribution is a group
on the sites of a lattice over the activity grid (`Group(on='sites')` on a one-dimensional `LatticeSpace`), with one
state per functional subsystem, so this change no longer adds a `Distribution` component. Relaxation toward an
equilibrium is a plain `Rate` law, and macro-scale external actions are rates of reduction expressions, so this change
adds no `Relaxation` or `ExternalAction` primitive; the name `Relaxation` is already the core's threshold law.
Observables are given to simulations and catalog entries, so moments are reduction expressions rather than system
observables. What this change still adds: the encounter laws and their gain-loss lowering, the moment reductions, the
particle solver with its declared support, the level conversions, and the library models.

## What Changes

- Add encounter laws to `core.laws`: encounters between a candidate, a field particle, and optionally a test particle
  happen at an encounter rate and transfer the candidate's activity with a transition probability. Proliferative and
  destructive encounters create and remove particles. Bellomo's gain and loss terms are derived from these and
  compiled to ODEs on the activity grid. The laws form a new law kind `encounter` of the continuous class, with a
  `Requires` declaration, and validation reports transition probabilities that do not sum to one.
- External actions are expressed with the general primitives: micro-scale actions as encounters whose field role binds
  a fixed distribution, and macro-scale actions as rate laws of the distribution's moments.
- Add moment reductions of site groups over an activity grid: density, mean activity, variance, and higher moments,
  usable in laws and as observables of simulations and catalog entries.
- Add a stochastic particle solver to `core.solvers` as an alternative for the same declarations, with a `Supports`
  declaration. It samples encounters between finite particles by direct simulation Monte Carlo, and agrees with the
  ODE solution as the particle number grows.
- Add level conversions: build a distribution from the states of a group's members, and seed a variable group's
  members from a distribution.
- Add library models: wealth and welfare dynamics, criminality against security forces, tumour-immune competition,
  and opinion formation with a discrete activity. Field-specific transition tables live in the model packages.

## Capabilities

### New Capabilities

- `kinetic-models`: encounter laws, moments, the particle solver, and level conversions over distributions held as
  site groups.

### Modified Capabilities

- `model-library`: kinetic models of active particles.

## Impact

- `skcomplex.core.laws` gains `Encounter`, `Proliferation`, and `Destruction`, and `skcomplex.core.base.LawKind` gains
  `encounter`. The moment reduction `moment` joins the existing reductions, and `skcomplex.core.components` gains the
  level conversions `distribution_from` and `members_from`. `skcomplex.core.solvers` gains `ParticleKinetics`.
- Builds on `redesign-system-model` for site groups, lattice spaces, observables on simulations, and validation, on
  `add-adaptive-networks` for distributions on network nodes, and on the experiments package for the convergence
  ensembles.
