# Spec Delta

## Purpose

Lets a user write field laws that read beyond the nearest neighbours: convolutions of field expressions with kernels,
reads at shifted and state-dependent positions, and the inversion of modified-Helmholtz (screened Poisson) equations,
all symbolic in the laws and lowered for every discretization that supports them.

## ADDED Requirements

### Requirement: Fields convolve with kernels
Laws of fields, groups on the sites of a lattice, SHALL convolve an expression of the field's states with a kernel written as a symbolic
function of the displacement, whose free symbols are the displacement and parameters of the model. The convolution
SHALL approximate the integral over the domain by the sum over sites weighted by the cell volume, and a kernel MAY
declare a radius beyond which it is zero.

#### Scenario: Gaussian smoothing
- **WHEN** a Gaussian profile of width 1 on a periodic line of length 40 and 400 sites is convolved with a normalised
  Gaussian kernel of width 0.5
- **THEN** the result matches the Gaussian of width `sqrt(1.25)` scaled by `1 / sqrt(1.25)` within 1e-8 at every site

#### Scenario: Kernel that reads a state
- **WHEN** a kernel's expression contains a state of the field
- **THEN** validation reports an issue whose path is the law and whose message names the state, before compilation

### Requirement: Convolutions respect the lattice edges
On periodic lattices a convolution SHALL wrap around. At fixed or zero-flux edges the integral SHALL cover the lattice
only, with the field read as its fixed value or its mirror image outside.

#### Scenario: Truncated domain
- **WHEN** a field with fixed zero edges on a 2D lattice of 64 × 48 sites is convolved with a kernel of radius 5 sites
- **THEN** the result equals the direct sum over the sites within the radius, times the cell area, within 1e-12

### Requirement: Kernels follow parameter changes
A convolution SHALL use the current values of the kernel's parameters, so a changed parameter between runs or across
the members of an experiment changes the convolution without the user resampling the kernel.

#### Scenario: Wider kernel
- **WHEN** a model is run twice with the kernel width parameter set to 1 and then to 2
- **THEN** the second convolution of the same field equals a direct convolution with the kernel of width 2

### Requirement: Laws read fields at shifted positions
Laws of fields on lattices SHALL read an expression of the field's states at a physical displacement per axis from
every site. A displacement that is a whole number of sites SHALL read the value at that site exactly. Any other
displacement, including one that depends on the states, SHALL interpolate at the shifted position, at least linearly.

#### Scenario: Whole-site shift
- **WHEN** a law reads a field at a displacement of three spacings along the first axis of a periodic lattice
- **THEN** the value at every site equals the field three sites ahead, exactly

#### Scenario: Look-ahead interpolation
- **WHEN** a sine wave on a periodic line of `n` sites is read at a displacement of 0.37 spacings plus a term
  proportional to a second state, for `n` equal to 100 and 200
- **THEN** the interpolation error against the exact shifted sine falls at least as fast as `1 / n²`

### Requirement: Shifted reads follow the boundary conditions
Positions that a shifted read places outside the lattice SHALL follow the field's boundary condition: periodic edges
wrap, zero-flux edges mirror, and fixed edges read the fixed value.

#### Scenario: Look-ahead beyond a fixed edge
- **WHEN** a field with the fixed edge value 0.2 is read beyond its last site
- **THEN** the read returns 0.2

### Requirement: Fields invert modified-Helmholtz operators
Laws SHALL use the solution `psi` of `(laplacian - screening²) psi = expression` of an expression of a field's
states, where the screening is a non-negative expression of the model's parameters. With zero screening the operator
SHALL equal the inverse Laplacian.

#### Scenario: Single Fourier mode
- **WHEN** `sin(k x) sin(l y)` on a periodic spectral field is inverted with screening `kappa`
- **THEN** the result equals `-sin(k x) sin(l y) / (k² + l² + kappa²)` within 1e-12

#### Scenario: Negative screening
- **WHEN** a law inverts with a screening whose square can be negative, or that depends on a state
- **THEN** validation reports an issue whose path is the law and whose message names the screening

### Requirement: Inversion runs on every discretization
The inversion SHALL be available on spectral fields, on finite-difference fields on lattices with periodic,
zero-flux, or fixed edges, on networks with the graph Laplacian, and on finite-element meshes. The solution SHALL
satisfy the field's boundary condition, or a boundary condition given to the operator. Where the Poisson equation
fixes the solution only up to a constant, on periodic or zero-flux edges and on networks, the solution SHALL have
zero mean.

#### Scenario: Closed basin
- **WHEN** a manufactured right-hand side on a finite-difference lattice with fixed zero edges is inverted with
  screening 2 at 32 and 64 sites per side
- **THEN** the solution is zero on the walls, and its error against the exact solution falls at second order

#### Scenario: Unscreened periodic inversion
- **WHEN** a field with nonzero mean on a periodic finite-difference lattice is inverted with zero screening
- **THEN** the solution has zero mean, and its Laplacian equals the field minus its mean within 1e-10

### Requirement: Nonlocal operators compose with local ones
The value of a convolution, a shifted read, or an inversion SHALL be usable inside the other operators, so that a
gradient, an advection, a Laplacian, or a shifted read of a nonlocal value reads that value at the neighbouring
sites. Nonlocal operators SHALL keep the field's discretization: on spectral fields chains of Fourier multipliers
SHALL combine into one multiplier.

#### Scenario: Velocity from vorticity in a basin
- **WHEN** a finite-difference field with fixed zero edges computes `gradient(inverse_helmholtz(q, kappa))` for a
  manufactured `q` at two resolutions
- **THEN** the gradient converges at second order to the exact gradient of the solution

### Requirement: Unsupported combinations fail clearly
Each nonlocal operator SHALL declare the space kinds it requires: a lattice for convolutions and shifted reads, and a
lattice, network, or mesh for inversions. Validation SHALL report, before compilation and with the law's path, the
operator, and a fix, every use where an operator has no meaning: a convolution or a shifted read on a network or a mesh,
a spectral convolution or inversion on a lattice without periodic edges, or an operator whose operand reads groups of
more than one space.

#### Scenario: Convolution on a network
- **WHEN** a law of a site group on a network convolves a state with a kernel
- **THEN** validation reports an issue that names the law and says the convolution needs a lattice space

#### Scenario: All misuses at once
- **WHEN** one system convolves on a network and inverts with a negative screening in two laws
- **THEN** `Simulation` raises one `ValidationError` that lists both issues, each with its law's path
