## Why

The Monte Carlo solver of the core moves one site at a time, so near a critical point its chains decorrelate slowly.
Normalizing flows and flow matching learn a map from a simple density to the Boltzmann density of a system. Used as
proposals of whole configurations with importance weights, they give independent samples, unbiased estimates, the
free energy, and an effective sample size that says how good the map is (Boltzmann generators, 2406.14426, and flows for
lattice field theory, 2504.18126). The core has the energies, the samplers, and JAX, but no proposal of a whole
configuration and no learned sampler.

## What Changes

- `SamplingInteraction` takes an independence proposal of whole configurations as its move, a new spec
  `ProposalSpec(sample, log_density, hits)`, and accepts it by the independence Metropolis rule. It then needs the new
  setting `energy`, the energy of one site whose sum over the sites is the total objective, since a global move cannot
  be scored by the local objective.
- New function `generative_sampler(system, *, kind=..., training=..., ...)` in `core/analysis`. It trains a coupling or
  spline flow (flowjax) or a flow-matching model (Equinox and Diffrax) on the energy of a Monte Carlo system or on its
  samples, on the shared Optax loop of `add-learning-foundation`, and returns a frozen result with the importance weights, the effective sample size, the free energy, the
  reweighted estimates of given observables, the trained sampler, and a `system` whose sampling interaction proposes
  from the trained sampler, which `simulate`, `sweep`, and the analyses accept.
- New system `LatticePhi4System` (`physics/statistical_mechanics`), the real scalar φ⁴ field theory on a periodic lattice
  of any dimension, sampled by local Gaussian moves, with the energy that flow proposals need.
- New runtime dependency `flowjax`. Equinox comes with `add-learning-foundation`.

## Capabilities

### New Capabilities

- `generative-samplers`: learned samplers of Boltzmann densities of Monte Carlo systems, their importance weights,
  effective sample sizes, free energies, and the systems that propose from them.

### Modified Capabilities

- `lattice-dynamics`: Monte Carlo sampling accepts independence proposals of whole configurations scored by a total
  energy.
- `model-library`: the lattice φ⁴ field theory, with its inventory entry and gallery example.

## Impact

- Core: `skcomplex.core.specs` (new `ProposalSpec`), `SamplingInteraction` and `MonteCarloKernel` (the global move and
  the total energy), and a new module of `skcomplex.core.analysis`. Existing moves and systems do not change.
- Systems: `skcomplex.systems.physics.statistical_mechanics`, with a new self-contained module.
- Dependencies: `flowjax` 19.2.1, released 2026-11-07. Equinox comes with `add-learning-foundation`, and jax, optax,
  and diffrax are already runtime dependencies.
- Docs: the lattice systems guide, the analysis guide, a gallery example, and the inventory with the primitive
  `generative-samplers`.
- Depends on `add-learning-foundation`, whose Equinox dependency, shared Optax loop with validation and early stopping,
  and result convention every sampler uses. flowjax supplies only the flows, which are Equinox modules, and not their
  training. Independent of the other drafts that build on the foundation.
