# Proposal

## Why

Kinetic equations evolve a distribution `f(t, x, v)` over position and velocity, or over position and orientation.
Particles stream through space at their velocity, forces push them through velocity, and collisions redistribute them
in velocity at a fixed position (Bellomo et al. 2017 §2.2–2.4). The framework has fields over lattices, networks, and
meshes, but every axis of a field is a position. No field has velocity or orientation axes, no law reads the moments of
a distribution over those axes, and no law expresses a collision operator. So the inventory of
`organize-model-library` records these models as blocked on this change:
- the Boltzmann, BGK, Vlasov, and Enskog equations (Bellomo et al. 2017 §2.2–2.3);
- the Bertin-Droz-Grégoire kinetic theory of flocking (Marchetti et al. 2013, arXiv 1207.2929 §V.2.3).

The active change `add-kinetic-models` covers distributions over a discrete activity variable, without position, and
lists spatial kinetic models with velocity and position as a non-goal. This change fills that gap and nothing else.

## What Changes

- Add phase spaces: the product of a position lattice and velocity or orientation axes, with uniform velocity grids and
  periodic orientation grids. A field over a phase space holds one value per point of the product.
- Add phase coordinates as symbols in laws, so a law can read the velocity or the orientation of the point it updates.
- Add transport operators: free streaming `v · ∇x f` along the position axes and acceleration `F · ∇v f` along the
  velocity axes, lowered to the existing upwind, finite-volume, and spectral discretizations.
- Add velocity moments as expressions in laws and as observables: density, mean velocity or polarization, and
  temperature or nematic order, as fields over the position lattice.
- Add collision operators: BGK relaxation toward the local Maxwellian with exact discrete conservation, the Boltzmann
  collision operator by the fast Fourier spectral method with an optional Enskog density factor, and angular collision
  and self-diffusion operators on orientation axes.
- Add mean-field coupling: a phase-space law reads a field over the position lattice, such as an electric field solved
  from a moment by the Poisson equation.
- Add a semi-Lagrangian transport stepper and Strang splitting of transport and collisions.
- Stream phase-space slices and moments to Rerun as images.
- Add library models: the Boltzmann, BGK, Vlasov-Poisson, and local Enskog equations, and Bertin-Droz-Grégoire
  kinetic flocking.

## Capabilities

### New Capabilities

- `phase-space-fields`: phase spaces, phase coordinates, transport operators, velocity moments, collision operators,
  mean-field coupling, the semi-Lagrangian and split steppers, and their visualization.

### Modified Capabilities

- `model-library`: kinetic equations in phase space.

## Impact

- `skcomplex.core.model` gains `PhaseSpace`, `VelocityAxis`, `OrientationAxis`, `streaming`, `acceleration`,
  `moment`, `BGK`, `Collision`, and `AngularCollision`. `Field` accepts a `PhaseSpace`.
- `skcomplex.core.compile` lowers the new operators and moments to NumPy and Numba, and `skcomplex.core.solvers` gains
  `SemiLagrangian` and `StrangSplitting`.
- No new runtime dependency: transport and collisions use `scipy.fft` and `scipy.ndimage`, which the project already
  declares. The design records the libraries surveyed and why each was not wrapped.
- `skcomplex.models.physics.fluids` and `skcomplex.models.physics.active_matter` gain the models, and the inventory
  moves them from `blocked add-phase-space-fields` to implemented.
- Does not depend on `add-kinetic-models`, and shares no name with it. The Enskog shift of the collision partners'
  positions waits for `add-nonlocal-operators`.
