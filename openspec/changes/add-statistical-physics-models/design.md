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
- **Couplings on lattices:** Monte Carlo laws read neighbours of the moved state only, so every site of
  `edwards_anderson` and `mattis_spin_glass` stores the couplings of its four bonds, and each bond is stored on both of
  its sites with the same value.
- **Per-link hops:** a plain role cannot bind a group member, so every link of `zero_range_process`,
  `dissipative_zero_range`, and `asep` binds its two sites of a lattice group through a variadic role.
  `neutral_clustering` writes all links as one interaction with a variadic role and one jump per directed link, which
  compiles about ten times faster than one interaction per link. These models run on the Numba backend by default.
- **ASEP rates:** the exit rate is named `gamma`, as in the book.
- **Biased trap walk:** the book's walk is in discrete time with deterministic trapping, so it is a deterministic
  lattice rule with one quenched landscape per row and the trapping times as constant lattice states.
- **Shaken spring-block chain:** a block sticks when its speed is below the numeric threshold `v_stick`, which must
  exceed the speed change of one solver step under friction.
- **Binary collision flocking:** a `Jump` has fixed increments and no pair events exist, so the continuous-time model is
  an agent rule that advances one unit of time in short substeps of self-diffusion kicks and binary collisions. It
  approaches the book's model as the substep shrinks.
- **Traffic density equation:** the book's closed equation (4.31) drops `dw/dt` when it eliminates the flux `w`, though
  that term is of the same order as the terms it keeps, and the closed equation is ill-posed in the unstable band. The
  model integrates the moment equations (4.26) and (4.27), whose homogeneous flow is unstable when
  `D(rho) < rho^2 u'(rho)^2 / gamma`.
- **Oscillator death with a quartic frequency density:** the threshold that follows from the book's quadratic (D.133)
  is `omega_0c = (1 + sqrt(2K - 1)) / sqrt 2`, not Eq. 7.62, and finite-N eigenvalues confirm it.
- **Heterogeneous global coupling:** the book leaves `g` and `h` general, and the model uses the Hopf normal form with
  time scales spread evenly around a mean of 1.
- **Doubling map:** floating-point doubling sends every orbit to 0 within about 53 steps, so points live on fractions
  `k/q` with a safe prime `q` below `2^50`, and the map is exact on them.
- **Degree-class SIR:** with the book's `k - 1` correction the epidemic threshold is `<k>/(<k^2> - <k>)`.
- **Origin-fixation walk:** the mutation rate is per locus.
- **Standardized agents:** the model has two marked configurations, so the standardization temperature is
  `K / (2 ln(H/2))`.
- **Diluted Hopfield network:** the asynchronous Glauber variant is expressible now. The book's parallel stochastic
  update waits for `add-stochastic-maps`, so the model is recorded as pending that change.

## Risks / Trade-offs

- **Per-link hopping is slow for large systems.** Gallery and test sizes stay small, and the later conserved-hopping
  change would remove the cost.
