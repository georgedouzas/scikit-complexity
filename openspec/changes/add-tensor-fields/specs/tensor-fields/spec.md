# Spec Delta

## Purpose

Lets a user write fields whose states are vectors or second-rank tensors with declared symmetries, apply tensor
calculus and objective time derivatives to them, and couple them to incompressible flow driven by a force density.

## ADDED Requirements

### Requirement: Tensor states keep their symmetry
The modelling layer SHALL declare states of rank 1 or 2 on fields, with a symmetry of `'symmetric'`, `'antisymmetric'`,
or `'symmetric-traceless'` for rank 2. The framework SHALL store only the independent components and SHALL give the
laws the full vector or matrix, with every dependent entry written in terms of the independent ones.

#### Scenario: Symmetric traceless tensor in two dimensions
- **WHEN** a user declares a symmetric traceless state `Q` on a 2D lattice of 32 × 32 sites
- **THEN** the field stores 2 components per site, and the trace of `Q` in the laws simplifies to exactly 0

#### Scenario: Symmetric traceless tensor in three dimensions
- **WHEN** a user declares a symmetric traceless state `Q` on a 3D lattice
- **THEN** the field stores 5 components per site, and `Q[1, 0]` is the same symbol as `Q[0, 1]`

#### Scenario: Law that breaks the symmetry
- **WHEN** a rate law on a symmetric traceless state has a right-hand side whose trace is not zero
- **THEN** compiling raises a named error that names the state and the trace

### Requirement: Tensor calculus acts on fields
Laws SHALL use the gradient of a vector, the divergence of a vector or a rank-2 tensor, the Laplacian of a vector or a
tensor, the strain-rate and vorticity tensors of a velocity, and the products, transposes, traces, and contractions of
tensor expressions. Every operator SHALL keep its result symbolic and SHALL lower to the field's discretization.

#### Scenario: Divergence of a tensor
- **WHEN** the divergence of the tensor `sin(x) I` is evaluated spectrally on a periodic 2D lattice of 64 × 64 sites
  over a period of 2π
- **THEN** the result equals `(cos(x), 0)` within 1e-10 at every site

#### Scenario: Strain rate and vorticity split the velocity gradient
- **WHEN** a user forms the strain-rate and vorticity tensors of a symbolic velocity
- **THEN** the strain rate is symmetric, the vorticity tensor is antisymmetric, and their sum simplifies to the
  velocity gradient

### Requirement: Objective derivatives transport tensors with the flow
The modelling layer SHALL provide the co-rotational derivative of a tensor or a vector with a flow-alignment parameter,
which advects it, rotates it with the vorticity, and aligns it with the strain rate, and the upper-convected
derivative of a rank-2 tensor.

#### Scenario: Upper-convected derivative of the identity
- **WHEN** the upper-convected derivative of the constant identity tensor is formed in a symbolic velocity
- **THEN** it simplifies to minus twice the strain-rate tensor

#### Scenario: Rotation by a uniform vorticity
- **WHEN** a uniform symmetric traceless tensor obeys a zero co-rotational derivative with alignment parameter 0 in a
  flow of uniform vorticity `ω`
- **THEN** its principal axis rotates at the angular velocity `ω / 2` within 1e-6 relative error

### Requirement: Incompressible velocities come from force densities
The modelling layer SHALL give a divergence-free velocity from a force density, in two and three dimensions, as the
overdamped Stokes velocity with a viscosity and an optional substrate friction, and as the projection of a velocity rate
onto divergence-free fields for inertial flow. The velocity SHALL be usable in any law of the same field.

#### Scenario: Divergence-free Stokes velocity
- **WHEN** the Stokes velocity of a random smooth force density is evaluated spectrally on a periodic 3D lattice
- **THEN** its discrete divergence is below 1e-10 times its largest component

#### Scenario: Stokes flow between walls
- **WHEN** the Stokes velocity of a uniform force along a channel is evaluated with no-slip walls on a bounded lattice
- **THEN** the profile across the channel matches the Poiseuille parabola within 1% of its peak

### Requirement: Tensor states take boundary conditions
Tensor states SHALL take periodic, zero-flux, or fixed boundary conditions per state, where a fixed value is a vector
or a tensor of the state's symmetry.

#### Scenario: Anchoring at walls
- **WHEN** a symmetric traceless state relaxes by elastic diffusion between walls with fixed tensors of perpendicular
  axes
- **THEN** the steady principal axis rotates linearly across the gap, within 1% of the angle difference

### Requirement: Principal axes and defects are analysed
The analysis layer SHALL return the principal axis and the scalar order of a symmetric tensor field at every site, and
the point defects of an axis field in two dimensions with their topological charges, for axes with or without head-tail
symmetry.

#### Scenario: Defect pair
- **WHEN** a nematic axis field holds one `+1/2` and one `-1/2` defect on a periodic lattice
- **THEN** the analysis finds exactly two defects at their sites within one lattice spacing, with charges `+1/2` and
  `-1/2`

### Requirement: Director fields stream to Rerun
Vector fields and the principal axes of tensor fields SHALL stream to Rerun as arrows or as headless line segments
over an image of a scalar, such as the scalar order.

#### Scenario: Nematic in the viewer
- **WHEN** a simulation of a symmetric traceless field streams to Rerun with the director visual
- **THEN** the viewer shows one segment per sampled site, along the principal axis, over an image of the scalar order,
  updated every frame
