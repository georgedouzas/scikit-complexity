## Why

Particle systems are expensive to run, and their laws are often unknown or too detailed. Graph networks learn a
one-step update of a particle system from its trajectories and roll it out much faster (LagrangeBench, 2309.16342).
Symplectic networks learn a many-body Hamiltonian whose rollout conserves energy (SympGNNs, 2408.16698). Force matching
fits the potential of coarse-grained beads to the mapped forces of a finer model (2310.18278). The core can run particle
groups with pair potentials and measure their neighbours, but it cannot learn a particle system, map particles to beads,
or compare two particle sets as distributions, which is how the learned rollouts are judged.

## What Changes

- New function `graph_network(source, ...)` in `core/analysis`. It trains an equinox encoder-processor-decoder graph
  network on particle trajectories, over radius graphs of every frame, to predict accelerations, and returns a frozen
  result whose `system` rolls the learned update out under discrete time.
- New function `hamiltonian_network(source, ...)` in `core/analysis`. It learns a separable Hamiltonian, the kinetic
  energy plus a permutation- and rotation-invariant network of pair distances, by matching symplectic steps to the
  trajectories, and returns a result whose `system` runs under continuous time with the symplectic solvers.
- New function `force_matching(source, forces=..., mapping=...)` in `core/analysis`. It maps particles to beads and fits
  a pair potential of the beads to the mapped forces, on a basis of radial functions by least squares or as an equinox
  network, and returns a result whose `system` runs the beads with the fitted potential.
- `coarse_grain` maps particle trajectories to bead trajectories with a mapping matrix or groups of particles, beside
  its grids and state densities.
- New function `distribution_distance(source, reference, kind=...)` in `core/analysis`, the maximum mean discrepancy,
  the Sinkhorn divergence, or the exact Wasserstein distance between two particle sets at every frame.
- New system `SphDamBreakSystem` (`physics/fluids`), a weakly compressible smoothed-particle-hydrodynamics dam break
  with a free surface and wall particles, the benchmark of 2309.16342.
- New runtime dependency `equinox`, shared with `add-learned-indicators`, `add-hybrid-neural-laws`, and
  `add-generative-samplers`, added by whichever lands first.

## Capabilities

### New Capabilities

- `particle-surrogates`: learned particle systems, by graph networks, symplectic Hamiltonian networks, and force
  matching of coarse-grained beads, each returned as a runnable system.

### Modified Capabilities

- `agent-analysis`: particle trajectories coarse-grained to beads, and distribution distances between particle sets.
- `model-library`: the SPH dam break, with its inventory entry and gallery example.

## Impact

- Core: four new modules of `skcomplex.core.analysis`, an extension of `coarse_grain`, and a private radius-graph
  helper in `core/analysis/_base.py`. No change to the engine.
- Systems: `skcomplex.systems.physics.fluids`, with a new self-contained module.
- Dependencies: `equinox` 0.13.8, released 2026-05-05. jax and optax are already runtime dependencies. No torch.
- Docs: the analysis guide, a gallery example, and the inventory with the primitive `particle-surrogates`.
- Independent of the other changes of this wave. `add-torch-surrogates` later adds torch-based force fields on top of
  `force_matching`.
