# Proposal

## Why

Marchetti et al. (2013, arXiv 1207.2929) describe active liquid crystals by an order parameter that is a tensor or a
vector at every point. A dry active nematic carries a density and the symmetric traceless alignment tensor `Q`, whose
divergence drives an active current (§II.2). Active nematic and polar gels couple `Q` or a polarisation vector `p` to an
incompressible flow, with an active stress `ζ Q` or `ζ p p`, and transport the order parameter with a co-rotational
derivative that rotates it with the local vorticity and aligns it with the strain rate (§III).

Site groups on lattices hold scalar and vector states, and the only incompressible flow is the two-dimensional
vorticity-streamfunction form of one scalar vorticity. A user can split `Q` into components by hand, but the framework
does not store or expose a tensor state by its symmetry, has no divergence of a tensor, no strain-rate or vorticity
tensor, no co-rotational or upper-convected derivative, and no way to get a divergence-free velocity from a force
density. The organisation of the model library records these models as blocked on this change.

**Rebased on `redesign-system-model`.** The redesign's inventory gate already added the tensor state domain,
`state(domain=Tensor(rank, symmetry))` with `DomainKind.TENSOR` and the symmetries `'symmetric'`, `'antisymmetric'`, and
`'traceless'`, as a declaration that validation reads, and it leaves fields of tensors to this change. This change
therefore no longer adds the `rank` and `symmetry` keywords of `state`: it gives the existing tensor domain its storage
of independent components, its symbolic matrix view, and its symmetry check, and adds the tensor calculus, objective
derivatives, incompressible velocities, boundaries, analysis, visual, and models. Tensor states live in site groups on a
lattice, `Group(on='sites')`, as every former `Field` does. The visual is named for what it draws, `AxisVisual`, rather
than the liquid-crystal term director, so the core stays application agnostic.

## What Changes

- Give the tensor domain storage and a symbolic view: a state of rank 1 or 2 per site with a declared symmetry,
  `'symmetric'`, `'antisymmetric'`, or `'traceless'` (symmetric and traceless). The framework stores only the
  independent components and exposes the state to laws as a symbolic vector or matrix whose dependent entries are
  written in terms of them.
- Add tensor calculus on site groups in `core.laws`: the gradient of a vector, the divergence of a tensor, the Laplacian
  of a vector or a tensor, the strain-rate and vorticity tensors of a velocity, and products and contractions of tensor
  expressions.
- Add objective time derivatives in `core.laws`: the co-rotational derivative with a flow-alignment parameter, as in the
  Beris-Edwards equations, and the upper-convected derivative.
- Add incompressible velocities from a force density in `core.laws`: the overdamped Stokes velocity with viscosity and
  optional substrate friction, and the Leray projection of a velocity rate for inertial flow, in two and three
  dimensions.
- Add boundary conditions for tensor states, with a fixed tensor value for anchoring at walls.
- Add analysis of tensor fields: the principal axis and the scalar order of a symmetric tensor, and the point defects
  of an axis field with their topological charges.
- Add an axis visual that streams the principal axis of a tensor state or a vector state to Rerun as line segments or
  arrows.
- Declare the tensor support of the solvers and backends, and validate tensor states: a tensor state outside the sites
  of a lattice, a law that breaks the declared symmetry, and a solver or backend without tensor support are reported by
  `validate()` before compilation.
- Add library models in `physics/active_matter`: the dry active nematic, the active nematic gel, and the active polar
  gel, with the Landau-de Gennes molecular field in their own package.
- Add `py-pde` as a runtime dependency for tensor operators and Poisson solves on bounded lattices.

## Capabilities

### New Capabilities

- `tensor-fields`: storage and views of tensor states with symmetries, tensor calculus, objective derivatives,
  incompressible velocities from forces, tensor boundary conditions, principal axes and defects, the axis visual, and
  their validation declarations.

### Modified Capabilities

- `model-library`: active nematic and polar models in `physics/active_matter`.

## Impact

- `skcomplex.core.components` gives the existing `Tensor` domain its expansion into independent components and its
  symbolic views. `skcomplex.core.laws` gains `vector_gradient`, `tensor_divergence`, `strain_rate`,
  `vorticity_tensor`, `corotational`, `upper_convected`, `stokes_velocity`, and `project`. `skcomplex.core.analysis`
  gains `principal_axis` and `axis_defects`. `skcomplex.core.viz` gains `AxisVisual`.
- `skcomplex.core.compile` lowers the new operators to the existing stencils and spectral multipliers, and a private
  adapter of the NumPy and Numba backends in `skcomplex.core.backends` lowers them to `py-pde` operators and Poisson
  solves on bounded finite-difference lattices.
- The ODE and stiff solvers and the NumPy and Numba backends add `DomainKind.TENSOR` to their declared support.
- `pyproject.toml` gains `py-pde` 0.58 or later. Its own dependencies, NumPy, SciPy, Numba, SymPy, matplotlib, and
  `tqdm`, are already installed.
- `skcomplex.models.physics.active_matter` gains three factories, and the inventory moves them from blocked to
  implemented.
