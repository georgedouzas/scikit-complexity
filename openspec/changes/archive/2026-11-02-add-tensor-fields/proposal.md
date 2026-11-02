# Proposal

## Why

Marchetti et al. (2013, arXiv 1207.2929) describe active liquid crystals by an order parameter that is a tensor or a
vector at every point. A dry active nematic carries a density and the symmetric traceless alignment tensor `Q`, whose
divergence drives an active current (§II.2). Active nematic and polar gels couple `Q` or a polarisation vector `p` to an
incompressible flow, with an active stress `ζ Q` or `ζ p p`, and transport the order parameter with a co-rotational
derivative that rotates it with the local vorticity and aligns it with the strain rate (§III). The inventory records
`dry_active_nematics` and `active_gels` in `physics/active_matter` as blocked on this change.

The current core already gives most of the numerics. A `FieldElement` holds one value per site for every state of its
element, `StateSpec(vector=True)` gives a state one component per dimension, and the law utilities `partial`,
`gradient`, `divergence`, `advection`, `laplacian`, and `inverse_laplacian` of `skcomplex.core.laws.utils` lower to
stencils with per-state boundaries, to spectral multipliers, and to finite volumes.
`GradientInteraction(variational=True)` gives the gradient flow of a free energy, `RateInteraction` adds any rate,
the `symmetry` setting of `FieldElement` gives walls on periodic lattices, and `AlignmentAnalysis`, `ImageVisual`, and
`RerunObserver` analyse and draw fields.
`DomainKind.TENSOR` exists in `skcomplex.types` but nothing declares, stores, or checks a tensor state. This change
therefore adds only what is missing: rank-2 tensor states with a declared symmetry, the tensor operations and objective
derivatives as law utilities composed from the existing operators, the Leray projection, the principal axis and defect
analyses, the axis visual, and the two models.

## What Changes

- Add rank-2 tensor states: `StateSpec(tensor=symmetry)` with the symmetry `'full'`, `'symmetric'`, `'antisymmetric'`,
  or `'traceless'` (symmetric and traceless). The element stores only the independent components, as plain real
  component states named by index, such as `Q_xx` and `Q_xy`, and the namespace gives laws the full Sage matrix, with
  every dependent entry written in terms of the independent ones. `EntrySpec` reports `DomainKind.TENSOR` for them.
- Let the `boundary` mapping of `FieldElement` take a matrix for a tensor state, which it expands into the fixed values
  of the components, for anchoring at walls.
- Add tensor operations to `skcomplex.core.laws.utils`, each a composition of `partial` and the existing operators:
  `vector_gradient`, `tensor_divergence`, `symmetric_part`, `antisymmetric_part`, and the trace and contraction of
  matrices through Sage.
- Add the objective time derivatives `corotational` and `upper_convected` to `skcomplex.core.laws.utils`.
- Add `leray_projection` to `skcomplex.core.laws.utils`, the divergence-free part of a vector expression,
  `v - gradient(inverse_laplacian(divergence(v)))`, exact on spectral fields of periodic lattices and with the walls
  of the `symmetry` setting.
- Add the compile issue `tensor-symmetry`, raised by `PassCompiler` when a rate law on a tensor state breaks its
  declared symmetry.
- Add the kind `'principal-axis'` to `AlignmentAnalysis`, the principal axis and scalar order of a symmetric tensor
  field, and `DefectAnalysis` in `skcomplex.analysis`, the point defects of an axis field and their charges.
- Add `AxisVisual` to `skcomplex.engine.viz`, which streams the principal axis of a tensor state or a vector state to
  Rerun as segments or arrows over an image of a scalar.
- Add the models `dry_active_nematics` and `active_gels` to `physics/active_matter`, with the Landau-de Gennes free
  energy as `LandauDeGennesEnergyInteraction`, a thin subclass of `GradientInteraction` in that package.

## Capabilities

### New Capabilities

- `tensor-fields`: tensor states with symmetries, tensor operations, objective derivatives, the Leray projection,
  tensor boundaries, the symmetry check, principal axes and defects, and the axis visual.

### Modified Capabilities

- `model-library`: the dry active nematic and the active nematic and polar gels in `physics/active_matter`.

## Impact

- `skcomplex.specs.StateSpec` gains `tensor`, and `skcomplex.specs.EntrySpec` expands a tensor state into its
  components and reports `DomainKind.TENSOR`.
- `skcomplex.core.elements.FieldElement` accepts a matrix boundary value for a tensor state.
- `skcomplex.core.laws.utils` gains `vector_gradient`, `tensor_divergence`, `symmetric_part`, `antisymmetric_part`,
  `corotational`, `upper_convected`, and `leray_projection`.
- `skcomplex.engine.compile.PassCompiler` gains the issue `tensor-symmetry`.
- `skcomplex.analysis` gains `DefectAnalysis` and the kind `'principal-axis'` of `AlignmentAnalysis`.
- `skcomplex.engine.viz` gains `AxisVisual`.
- `skcomplex.models.physics.active_matter` gains `dry_active_nematics`, `active_gels`, and
  `LandauDeGennesEnergyInteraction`, and `docs/overview/inventory.toml` marks the two entries implemented.
- No new dependency. Solvers and backends need no new support, since tensor components are plain real states.
