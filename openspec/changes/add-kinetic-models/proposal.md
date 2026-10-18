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
also calls for when generating models at a chosen resolution. The framework has agents and aggregates, but no element
whose state is a distribution and no encounter law between distributions.

## What Changes

- Add distribution states: an element whose state is a distribution over a discrete grid of activity values, with one
  or more functional subsystems, optionally on the nodes of a network.
- Add encounter laws: encounters between a candidate, a field particle, and optionally a test particle happen at an
  encounter rate and transfer the candidate's activity with a transition probability. Proliferative and destructive
  terms create and remove particles, and relaxation terms pull the distribution toward an equilibrium.
  Bellomo's gain and loss terms are derived from these and compiled to ODEs on the activity grid.
- Add external actions at the micro scale, on transitions, and at the macro scale, on the moments.
- Add moments of distributions as observables: density, mean activity, variance, and higher moments.
- Add a stochastic particle solver as an alternative for the same declarations. It samples encounters between finite
  particles by direct simulation Monte Carlo, and agrees with the ODE solution as the particle number grows.
- Add level conversions: build a distribution from an agent group's states, and seed agents from a distribution.
- Add library models: wealth and welfare dynamics, criminality against security forces, tumour-immune competition,
  and opinion formation with a discrete activity.

## Capabilities

### New Capabilities

- `kinetic-models`: distribution states, encounter laws, external actions, moments, the particle solver, and level
  conversions.

### Modified Capabilities

- `model-library`: kinetic models of active particles.

## Impact

- `skcomplex.model` gains `Distribution`, `Encounter`, `Proliferation`, `Destruction`, `Relaxation`, and
  `ExternalAction`, and `skcomplex.solvers` gains `ParticleKinetics`.
- Depends on `add-general-model-interface` for observables, on `add-adaptive-networks` for distributions on network
  nodes, and on `add-experiments` for the convergence ensembles.
