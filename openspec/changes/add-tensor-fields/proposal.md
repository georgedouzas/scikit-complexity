# Proposal

## Why

Marchetti et al. (2013, arXiv 1207.2929) describe active liquid crystals by an order parameter that is a tensor or a
vector at every point. A dry active nematic carries a density and the symmetric traceless alignment tensor `Q`, whose
divergence drives an active current (§II.2). Active nematic and polar gels couple `Q` or a polarisation vector `p` to an
incompressible flow, with an active stress `ζ Q` or `ζ p p`, and transport the order parameter with a co-rotational
derivative that rotates it with the local vorticity and aligns it with the strain rate (§III).

Fields today hold scalar states, and the only incompressible flow is the two-dimensional vorticity-streamfunction form
of one scalar vorticity. A user can split `Q` into components by hand, but the framework does not know that `Q` is
symmetric and traceless, has no divergence of a tensor, no strain-rate or vorticity tensor, no co-rotational or
upper-convected derivative, and no way to get a divergence-free velocity from a force density. The organisation of the
model library records these models as blocked on this change.

## What Changes

- Add tensor-valued states: a state of rank 1 or 2 per site, with a declared symmetry, `'symmetric'`,
  `'antisymmetric'`, or `'symmetric-traceless'`. The framework stores only the independent components and exposes the
  state to laws as a symbolic vector or matrix whose dependent entries are written in terms of them.
- Add tensor calculus on fields: the gradient of a vector, the divergence of a tensor, the Laplacian of a vector or a
  tensor, the strain-rate and vorticity tensors of a velocity, and products and contractions of tensor expressions.
- Add objective time derivatives: the co-rotational derivative with a flow-alignment parameter, as in the
  Beris-Edwards equations, and the upper-convected derivative.
- Add incompressible velocities from a force density: the overdamped Stokes velocity with viscosity and optional
  substrate friction, and the Leray projection of a velocity rate for inertial flow, in two and three dimensions.
- Add boundary conditions for tensor states, with a fixed tensor value for anchoring at walls.
- Add analysis of tensor fields: the principal axis and the scalar order of a symmetric tensor, and the point defects
  of an axis field with their topological charges.
- Add a director visual that streams the principal axis of a tensor field or a vector field to Rerun as line segments
  or arrows.
- Add library models in `physics/active_matter`: the dry active nematic, the active nematic gel, and the active polar
  gel.
- Add `py-pde` as a runtime dependency for tensor operators and Poisson solves on bounded lattices.

## Capabilities

### New Capabilities

- `tensor-fields`: tensor-valued states with symmetries, tensor calculus, objective derivatives, incompressible
  velocities from forces, tensor boundary conditions, principal axes and defects, and the director visual.

### Modified Capabilities

- `model-library`: active nematic and polar models in `physics/active_matter`.

## Impact

- `skcomplex.core.base.state` gains the `rank` and `symmetry` keywords. `skcomplex.core.model` gains
  `vector_gradient`, `tensor_divergence`, `strain_rate`, `vorticity_tensor`, `corotational`, `upper_convected`,
  `stokes_velocity`, and `project`. `skcomplex.core.analysis` gains `principal_axis` and `axis_defects`.
  `skcomplex.core.viz` gains `DirectorVisual`.
- `skcomplex.core.compile` lowers the new operators to the existing stencils and spectral multipliers, and to `py-pde`
  operators and Poisson solves on bounded finite-difference lattices.
- `pyproject.toml` gains `py-pde` 0.58 or later. Its own dependencies, NumPy, SciPy, Numba, SymPy, matplotlib, and
  `tqdm`, are already installed.
- `skcomplex.models.physics.active_matter` gains three factories, and the inventory moves them from blocked to
  implemented.
