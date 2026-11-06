# Proposal

## Why

Colloids and microswimmers move through a viscous fluid at low Reynolds number, where the force on one particle moves
every other particle through the flow it creates. The core couples particles only through pair potentials with a
friction of their own, so it cannot express the correlated motion of two optically trapped beads (Argun et al. 2021,
Problems 7.4 and 7.5) or the hydrodynamic interactions of squirmers (2405.15751 §IV, after Lighthill 1952, Blake 1971,
and Ishikawa et al. 2006). The audit lists both as missing. The user chose to write the mobility tensors in the core
with NumPy and Numba rather than wrap PyStokes, whose last release dates from April 2024.

## What Changes

- Add `MobilityInteraction` to `skcomplex.core.components.interactions`, the linear map of a mobility tensor from the
  forces and torques on the members of a group to their velocities and angular velocities, with the Oseen and the
  Rotne-Prager-Yamakawa tensors and their translational and rotational couplings.
- The interaction also carries the flows of force-free sources, the stresslet and the potential dipole of every member,
  to the other members, with an intrinsic velocity of every member, so active slip couples through the same operator.
- Pair forces of a potential of the distance enter the mobility like external forces.
- A continuous form gives rate laws of the positions and orientations, and a discrete form gives the Ermak-McCammon
  step with Brownian displacements whose covariance is `2 kT M dt`, drawn by a Cholesky factor or a Lanczos (Krylov)
  square root of the mobility.
- Add the catalog system `squirmers` in `physics/active_matter`, spherical squirmers with the B1 and B2 slip modes as
  pushers or pullers, and `hydrodynamically_coupled_traps` in `physics/soft_matter`, two beads in blinking optical
  traps whose free motion shows the Oseen correlation, each with tests, inventory entries, and gallery examples.
- The Jacobian pattern of a call over a whole group couples every member, as the dense mobility needs.
- Document the interaction in the user guide.

## Capabilities

### New Capabilities

- `hydrodynamic-interactions`: mobility tensors of particles in a Stokes fluid, their active slip sources, and Brownian
  displacements consistent with them.

### Modified Capabilities

- `model-library`: the squirmer and the hydrodynamically coupled trap systems.
- `model-compilation`: the Jacobian pattern of calls over a whole group.

## Impact

- New module `src/skcomplex/core/components/interactions/_mobility.py`, exported from the package, and a dense
  pattern for vectorized calls in `src/skcomplex/core/engine/problems/_jacobian_kernel.py`. No new package and
  no new dependency, since NumPy and Numba are runtime dependencies already.
- New system modules in `skcomplex.systems.physics.active_matter` and `skcomplex.systems.physics.soft_matter`, their
  tests, inventory entries with a new `hydrodynamic-interactions` primitive, gallery examples, and a user guide section.
