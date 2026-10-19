# Design

## Context

The survey of Bertin 2021 checked every model against the current core:
- **Mixing laws:** a system that mixes `Jump` with `Rate` or `Noise` is split into co-simulated partitions.
- **Neighbour reads:** lattice reads take states only, so quenched couplings are stored as constant states.
- **Noisy fields:** fields with noise exist, as in `toner_tu`.
- **Particle hops:** a `Jump` changes fixed increments, so a particle hop between sites is one element per site and one
  interaction per link, as `binary_ant_decision` does.

## Goals / Non-Goals

**Goals:**
- Every Bertin model recorded in the inventory.
- The 31 implementable models built from public core names, with scenarios and gallery examples.

**Non-Goals:**
- **Analysis methods:** the book's analysis methods, such as large deviations, aging, and record statistics, are
  specified by `extend-analysis-toolkit`.
- **Conserved hopping:** conserved hopping as one group, which would make zero-range and exclusion processes faster.
  The per-link construction is faithful, and the survey found no mature library for it. A later change may add jumps
  addressed by member offsets.

## Decisions

- **Spin glasses:**
  - The Sherrington-Kirkpatrick and Curie-Weiss models use group jumps whose rates read the local field through a
    weighted reduction, `total(s, weights=J)`.
  - The Edwards-Anderson and Mattis models use lattice Monte Carlo with couplings as constant lattice states.
- **Flashing ratchet:** the switching of its potential is a pre-sampled Poisson series, seeded by the simulation's
  random state.
- **Shaken spring-block chain:** a driven noisy phase, then relaxation with Coulomb friction and stick events, as
  `burridge_knopoff` does.
- **Diluted Hopfield network:** the asynchronous Glauber variant is expressible now. The book's parallel stochastic
  update waits for `add-stochastic-maps`, so the model is recorded as pending that change.

## Risks / Trade-offs

- **Per-link hopping is slow for large systems.** Gallery and test sizes stay small, and the later conserved-hopping
  change would remove the cost.
