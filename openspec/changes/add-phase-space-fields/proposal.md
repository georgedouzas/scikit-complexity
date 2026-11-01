# Proposal

## Why

Kinetic equations evolve a distribution `f(t, x, v)` over position and velocity, or over position and orientation.
Particles stream through space at their velocity, forces push them through velocity, and collisions redistribute them
in velocity at a fixed position (Bellomo et al. 2017 §2.2–2.4). The model inventory records five models as blocked on
this change:
- the Boltzmann, BGK, Vlasov, and Enskog equations (Bellomo et al. 2017 §2.2–2.3),
- the Bertin-Droz-Grégoire kinetic theory of flocking (Marchetti et al. 2013, arXiv 1207.2929 §V.2.3).

The core already provides most of what these models are made of, so this change builds on it and drops it from its
scope:
- `DensityElement` is the value `f` of a density at one point of a grid, and `FieldElement` holds it at every site of
  a `LatticeSpace` with finite-difference, spectral, or finite-volume operators,
- `EncounterInteraction` gives bilinear encounters of densities over classes, conservative, destructive, or
  proliferative, with a rate `eta[h, k]` and a kernel `kernel[h, k, i]`,
- `AdvectionInteraction` carries a state by a drift in conservative form, `moment` reduces a group over a grid of
  coordinates, `RelaxationInteraction` relaxes a state towards a goal, and `inverse_laplacian` and `gradient` solve and
  differentiate on a position lattice.

What phase-space fields add beyond them is the product of position axes with class axes, so that a density depends on
both. Concretely the change adds:
- a lattice whose axes are not all positions, with a coordinate grid and a boundary per axis, so encounters and
  moments act along the class axes at every position while transport acts along the position axes,
- coordinates that laws read, so a drift along a position axis can be the velocity coordinate of each site,
- moments over the class axes only, which are fields over position,
- the binary collision integral on continuous velocity grids, where the dense kernel of `EncounterInteraction` would
  need `N³` entries for `N` velocity points, and the maximum-entropy equilibrium with the moments of a density,
- the solvers that kinetic equations need, a semi-Lagrangian transport step and Strang splitting of transport and
  collisions.

## What Changes

- Add `PhaseSpace(position, axes)` to `skcomplex.core.spaces`, a lattice space that is the product of a position
  `LatticeSpace` and class axes, each an `AxisSpec(name, size, low, high, periodic=False)` of `skcomplex.specs` with
  its own coordinate grid and boundary.
- Add the law utility `coordinate(name)`, the coordinate of the site a law updates along a named axis.
- Let `moment` take `axes`, reducing over the class axes of a phase space into a field over position that laws read
  and observables record.
- Let `EncounterInteraction` run at every position of a phase space, over its class axes, and take no `field`, for a
  linear redistribution at a rate `eta[h]` with a kernel `kernel[h, i]`.
- Add `CollisionInteraction(target, state, cross_section, factor=None)` to `skcomplex.core.interactions`, the binary
  collision integral of a density over the continuous class axes of a phase space, lowered to a new `CollisionLaw` by
  the fast spectral method with an optional factor of the local density.
- Add the law utility `maximum_entropy(expression, axes, order=2)`, the density on the class axes with the largest
  entropy whose moments up to the order equal those of the expression at every position.
- Add `sources` roles to `ConservationInteraction` and `AdvectionInteraction`, so a drift reads a field over the
  position lattice, broadcast along the class axes.
- Add `SemiLagrangianSolver` and `StrangSplittingSolver(transport, collision)` to `skcomplex.engine.solvers`, each
  with its `SupportSpec`.
- Report through the compile report a phase space larger than its size limit, a `coordinate` or a `moment` over axes
  in a law whose group is not on a `PhaseSpace`, and a collision law on a phase space without continuous class axes.
- Let `ImageVisual` take a `slice` of two axes and log moments over position, so a phase-space field streams to Rerun
  as its position-velocity image.
- Add library models built on these classes: the Boltzmann, BGK, Vlasov-Poisson, and local Enskog equations in
  `physics/fluids`, with `BhatnagarGrossKrookInteraction`, a `RelaxationInteraction` towards the maximum-entropy
  equilibrium, and the Bertin-Droz-Grégoire kinetic flocking in `physics/active_matter`, with angular alignment and
  angular diffusion as `EncounterInteraction` subclasses.

## Capabilities

### New Capabilities

- `phase-space-fields`: phase spaces, coordinates in laws, moments over class axes, per-position and linear
  encounters, the binary collision integral, maximum-entropy equilibria, transport of phase-space fields, the
  semi-Lagrangian and split solvers, and their visualization.

### Modified Capabilities

- `model-library`: kinetic equations in phase space.

## Impact

- `skcomplex.core.spaces` gains `PhaseSpace`, and `skcomplex.specs` gains `AxisSpec`. `skcomplex.core.laws.utils`
  gains `coordinate` and `maximum_entropy`, and `moment` gains `axes`. `skcomplex.core.laws` gains `CollisionLaw` with
  its `RequirementSpec`, and `skcomplex.core.interactions` gains `CollisionInteraction`. `EncounterInteraction` takes
  an optional `field`, and `ConservationInteraction` and `AdvectionInteraction` take `sources`.
- `skcomplex.engine.compile` and `skcomplex.engine.problems` lower coordinates, partial moments, per-position
  encounters, the maximum-entropy equilibrium, and collision kernels for the NumPy and Numba backends.
  `skcomplex.engine.solvers` gains `SemiLagrangianSolver` and `StrangSplittingSolver`, and `skcomplex.engine.viz`
  gains the `slice` and `moments` arguments of `ImageVisual`.
- No new runtime dependency: transport and collisions use `scipy.fft` and `scipy.ndimage`, which the project already
  declares. The design records the libraries surveyed and why each was not wrapped.
- `skcomplex.models.physics.fluids` and `skcomplex.models.physics.active_matter` gain the models, and the inventory
  moves them from `blocked add-phase-space-fields` to implemented.
- Shares with `adopt-pyclaw-finite-volumes` the source rates that `ProblemSpec` keeps apart from the flux differences,
  added by whichever change lands first. The Enskog shift of the collision partners' positions waits for
  `add-nonlocal-operators`.
