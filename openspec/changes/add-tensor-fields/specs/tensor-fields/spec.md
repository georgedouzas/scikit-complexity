# Spec Delta

## Purpose

Lets a user write fields whose states are second-rank tensors with declared symmetries, apply tensor operations and
objective time derivatives to them, and project vector fields onto divergence-free fields.

## ADDED Requirements

### Requirement: Tensor states keep their symmetry
`StateSpec` SHALL take a rank-2 tensor symmetry, `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`
(symmetric and traceless). The element SHALL store only the independent components as real component states, and the
namespace SHALL give the laws the full Sage matrix, with every dependent entry written in terms of the independent ones.
`EntrySpec` SHALL report the kind `DomainKind.TENSOR` for the components.

#### Scenario: Symmetric traceless tensor in two dimensions
- **WHEN** a user declares a state `Q` with `StateSpec(tensor='traceless')` in a `FieldElement` on a 32 × 32 lattice
- **THEN** the field stores 2 components per site, and the trace of `s.Q` in the laws simplifies to exactly 0

#### Scenario: Symmetric traceless tensor in three dimensions
- **WHEN** a user declares the same state on a 3D lattice
- **THEN** the field stores 5 components per site, and `s.Q[1, 0]` is the same expression as `s.Q[0, 1]`

#### Scenario: Vector and tensor together
- **WHEN** a user declares `StateSpec(vector=True, tensor='symmetric')`
- **THEN** checking the spec raises a `ValidationError` that names both settings

### Requirement: Rate laws keep the declared symmetry
`PassCompiler` SHALL report a rate law on a tensor state whose value breaks the declared symmetry, with the code
`tensor-symmetry`, before compilation.

#### Scenario: Law that breaks the symmetry
- **WHEN** a rate law on a symmetric traceless state has a value whose trace is not zero
- **THEN** the report of `PassCompiler` holds an issue with the code `tensor-symmetry` that names the law, the state,
  and the trace

### Requirement: Tensor operations act on fields
The law utilities SHALL give the gradient of a vector, the divergence of a rank-2 tensor, and the symmetric and
antisymmetric parts of a matrix, as expressions of the existing field operators, so every operation lowers to the
discretization of its field. A tensor utility given an expression of the wrong shape SHALL raise a `ValidationError`
that the compiler reports with the code `law-domain`.

#### Scenario: Divergence of a tensor
- **WHEN** `tensor_divergence` of the tensor `sin(x) I` is evaluated on a spectral field of a periodic 64 × 64 lattice
  over a period of 2π
- **THEN** the result equals `(cos(x), 0)` within 1e-10 at every site

#### Scenario: Strain rate and vorticity split the velocity gradient
- **WHEN** a user forms `symmetric_part` and `antisymmetric_part` of `vector_gradient` of a symbolic velocity
- **THEN** the first is symmetric, the second antisymmetric, and their sum simplifies to the velocity gradient

#### Scenario: Tensor divergence of a scalar
- **WHEN** a law applies `tensor_divergence` to a scalar state
- **THEN** the report of `PassCompiler` holds an issue with the code `law-domain` that names the utility and the state

### Requirement: Objective derivatives transport tensors with the flow
The law utilities SHALL give the co-rotational derivative of a symmetric tensor or a vector with a flow-alignment
parameter, which advects it, rotates it with the vorticity, and aligns it with the strain rate, and the upper-convected
derivative of a rank-2 tensor.

#### Scenario: Upper-convected derivative of the identity
- **WHEN** `upper_convected` of the constant identity tensor is formed in a symbolic velocity
- **THEN** it simplifies to minus twice the symmetric part of the velocity gradient

#### Scenario: Rotation by a uniform vorticity
- **WHEN** a uniform symmetric traceless tensor obeys a zero `corotational` derivative with alignment parameter 0 in a
  flow of uniform vorticity `ω`
- **THEN** its principal axis rotates at the angular velocity `ω / 2` within 1e-6 relative error

### Requirement: Vector fields project onto divergence-free fields
The law utilities SHALL give the Leray projection of a vector expression, its divergence-free part, built from the
existing `gradient`, `divergence`, and `inverse_laplacian`, in two and three dimensions on spectral fields of periodic
lattices, including the walls that the `symmetry` setting of `FieldElement` gives.

#### Scenario: Divergence-free projection
- **WHEN** `leray_projection` of a random smooth vector field is evaluated on a spectral field of a periodic 3D lattice
- **THEN** its discrete divergence is below 1e-10 times its largest component

#### Scenario: Stokes flow between walls
- **WHEN** the overdamped Stokes velocity `leray_projection(inverse_laplacian(-f / eta))` of a uniform force along a
  channel is evaluated with the velocity odd about both walls of a symmetry-extended periodic lattice
- **THEN** the profile across the channel matches the Poiseuille parabola within 1% of its peak

### Requirement: Tensor states take boundary values
The `boundary` mapping of `FieldElement` SHALL take a matrix of the state's symmetry as the fixed value of a tensor
state, and SHALL reject a matrix without that symmetry with a `ValidationError` that names the state.

#### Scenario: Anchoring at walls
- **WHEN** a symmetric traceless state relaxes by `laplacian` between fixed edges holding tensors of perpendicular axes
- **THEN** the steady principal axis rotates linearly across the gap, within 1% of the angle difference

### Requirement: Principal axes and defects are analysed
`AlignmentAnalysis` SHALL give, with the kind `'principal-axis'`, the principal axis and the scalar order of a
symmetric tensor field at every site. `DefectAnalysis` SHALL give the point defects of a two-dimensional axis field with
their topological charges, for axes with or without head-tail symmetry.

#### Scenario: Defect pair
- **WHEN** `DefectAnalysis(period=np.pi)` reads a nematic axis field with one `+1/2` and one `-1/2` defect on a periodic
  lattice
- **THEN** it finds exactly two defects within one lattice spacing of their sites, with charges `+1/2` and `-1/2`

#### Scenario: Polar vortex
- **WHEN** `DefectAnalysis(period=2 * np.pi)` reads a vector field with one vortex
- **THEN** it finds one defect of charge `+1`

### Requirement: Axis fields stream to Rerun
`AxisVisual` SHALL stream the principal axes of a tensor state as headless segments, or a vector state as arrows, at a
sampled subset of sites, over an image of a scalar state.

#### Scenario: Nematic in the viewer
- **WHEN** a simulation of a symmetric traceless state streams to Rerun with `AxisVisual`
- **THEN** the recording holds one segment per sampled site along the principal axis, over an image of the scalar, at
  every frame
