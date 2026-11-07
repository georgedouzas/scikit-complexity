# Design

## Context

Interactions return symbolic laws that the compiler lowers to NumPy and Numba. A pair interaction sums a symbolic law
over the pairs within a cutoff, and a noise law gives every component an amplitude of a Wiener process, with fixed
covariances only. A mobility tensor is dense: the velocity of a member depends on the forces on every member through a
matrix of the positions, and Brownian motion consistent with it needs a square root of that matrix at every step.
Neither fits a law written per member or per pair. A `CallSpec(..., vectorized=True)` value runs once per evaluation
with one array per argument over the whole group, and `random=True` passes the random stream of the run, so one numeric
function can compute the dense product and its noise for all members.

## Goals / Non-Goals

**Goals:**

- One core class that names the operation, the mobility, configured by functions of the namespace for the forces,
  torques, intrinsic velocities, stresslets, and potential dipoles of the members.
- Oseen and Rotne-Prager-Yamakawa tensors with all translational and rotational couplings, in NumPy and Numba.
- Brownian displacements of covariance `2 kT M dt` by Cholesky or Lanczos.
- Squirmers and the blinking traps as catalog systems that exercise it.

**Non-Goals:**

- Periodic boxes (Ewald sums), walls (Blake tensor), lubrication, and the exact two-sphere functions of Batchelor 1976.
- Spheres of different radii, and stresslets of rigid bodies induced by the strain of the flow.
- Brownian noise in continuous time, which would need a noise amplitude that is a dense matrix of the state.

## Decisions

- **Name.** `MobilityInteraction` in the existing package `skcomplex.core.components.interactions`, new module
  `_mobility.py`. The tensor is a variant string, `'oseen'` or `'rotne-prager-yamakawa'`, checked against a module
  constant. No new package.
- **Values through one vectorized call.** The laws are a rate or update law of the position and, when the members have
  an orientation, of the orientation, each a vectorized `CallSpec` whose arguments are the components of the positions,
  orientations, forces, torques, velocities, stresslets, dipoles, and the parameters. The two laws share one evaluation
  through a private cache keyed by the time and the arguments, so the orientation reads the same noise as the position.
  The alternative, `N d` noise laws per member over a `WienerProcessSpec`, needs the group size in the interaction and
  repeats the factorization per law.
- **Forms.** `form='rate'` gives `dx/dt = U` and `dp/dt = Omega x p` in continuous time without noise. `form='map'`
  gives the Ermak-McCammon step of the parameter `dt` in discrete time, `x + U dt + sqrt(2 kT dt) (B xi)_t`, and turns
  the orientation by `Omega dt + sqrt(2 kT dt) (B xi)_r`, normalized. The divergence of both tensors vanishes in an
  unbounded fluid, so the step needs no drift correction, and the normalization gives the Itô drift of rotational
  diffusion. In the map form the time of the namespace counts steps, so a force of the physical time reads `s.t * s.dt`.
- **Numerics.** Module-level Numba kernels give the 3x3 blocks of a pair, the matrix-free product with forces and
  torques, the dense grand mobility, and the flows of stresslets and dipoles, as the encounter kernel does. Everything
  is computed in three dimensions and planar positions embed in `z = 0`, keeping the in-plane translations and the
  rotation about `z`, a principal submatrix that stays positive definite. Cholesky factors the dense submatrix with
  NumPy. Lanczos runs the Krylov iteration of Chow and Saad (2014) on the matrix-free product until the relative change
  of the sample falls below `tolerance`, so it needs no dense matrix.
- **Active slip.** The stresslet `S` gives the flow `-3 (r.S.r) r / (8 pi eta r^5)` and the angular velocity
  `3 r x (S r) / (8 pi eta r^5)`, and the potential dipole `D` the flow `(3 r r / r^2 - I) D / r^3` and no rotation, at
  separations of a few radii. The squirmer system sets them from `B1` and `B2`, so the core never names a squirmer.
- **Jacobian pattern.** The pattern of a member kernel assumed that a call reads the arguments of its own member. A
  vectorized call reads the arguments of all members, so its rows become dense over the states its arguments read,
  which implicit solvers and the catalog check of the pattern need. The alternative of a dense Jacobian for the whole
  system would hide the sparsity of every other law.
- **Pair forces.** A `potential(namespace, r)` of the distance, with the parameters of the interaction in the
  namespace, is differentiated with Sage once and evaluated with `fast_callable` over the pairs within `cutoff`.

## Risks / Trade-offs

- [Dense cost] The product is `O(N^2)` and Cholesky `O(N^3)` per step → Lanczos is matrix-free, and the documentation
  states the cost, so a few hundred members are practical.
- [Oseen not positive at short range] → Cholesky failure raises a `ValidationError` that suggests the
  Rotne-Prager-Yamakawa tensor.
- [Far-field slip] Stresslet and dipole flows diverge at contact → the squirmer system adds a soft repulsion, and the
  documentation states the far-field range.
- [Python call per evaluation] The vectorized call runs in Python on every backend → the inner loops are Numba kernels.
