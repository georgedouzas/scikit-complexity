# nonlocal-operators Specification

## Purpose
Lets a user write field laws that read beyond the nearest neighbours: convolutions of field expressions with kernels,
reads at shifted and state-dependent positions, and the inversion of modified-Helmholtz (screened Poisson) equations,
all symbolic in the laws and lowered for every discretization that supports them, and lets a coupling over the
neighbours of a field lower on the field.

## Requirements

### Requirement: Fields convolve with kernels
Laws of a `FieldElement` on a lattice SHALL convolve an expression of the field's states with a kernel written as a
symbolic function of the displacement, whose free symbols are the displacement and parameters of the system. The
convolution SHALL approximate the integral over the domain by the sum over sites weighted by the cell volume, and a
kernel MAY declare a radius beyond which it is zero.

#### Scenario: Gaussian smoothing
- **WHEN** a Gaussian profile of width 1 on a periodic line of length 40 and 400 sites is convolved with a normalised
  Gaussian kernel of width 0.5
- **THEN** the result matches the Gaussian of width `sqrt(1.25)` scaled by `1 / sqrt(1.25)` within 1e-8 at every site

#### Scenario: Kernel that reads a state
- **WHEN** a kernel's expression contains a state of the field
- **THEN** the compiler's report holds an issue with the code `kernel-state` whose path is the law and whose message
  names the state

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
- **WHEN** a system is run twice with the kernel width parameter set to 1 and then to 2
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
states, where the screening is a non-negative expression of the system's parameters. With zero screening the operator
SHALL equal the inverse Laplacian.

#### Scenario: Single Fourier mode
- **WHEN** `sin(k x) sin(l y)` on a periodic spectral field is inverted with screening `kappa`
- **THEN** the result equals `-sin(k x) sin(l y) / (k² + l² + kappa²)` within 1e-12

#### Scenario: Negative screening
- **WHEN** a law inverts with a screening whose square can be negative, or that depends on a state
- **THEN** the compiler's report holds an issue with the code `screening` whose path is the law and whose message
  names the screening

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
The compiler SHALL report, in its report and with the law's path, the operator, and a fix, every use where a nonlocal
operator has no meaning: a convolution or a shifted read on a network or a mesh, a spectral convolution or inversion on
a lattice without periodic edges, or an operator whose operand reads fields of more than one space.

#### Scenario: Convolution on a network
- **WHEN** a law of a field on a network convolves a state with a kernel
- **THEN** the compiler's report holds an issue with the code `law-space` that names the law and says the convolution
  needs a lattice space

#### Scenario: All misuses at once
- **WHEN** one system convolves on a network and inverts with a negative screening in two laws
- **THEN** running the compiler raises one `ValidationError` that lists both issues, each with its law's path

### Requirement: Couplings lower on fields
`CouplingInteraction` with a `FieldElement` target SHALL lower its reduction over the neighbours of every site to
whole-site shifted reads on a lattice and to a gather over the neighbour table on a network, and SHALL lower a constant
times the difference of the ends to `laplacian`.

#### Scenario: Function of both ends on a lattice
- **WHEN** a coupling on a periodic lattice field sums `logistic(source.x - theta)` over the four nearest sites
- **THEN** the compiled rate equals a direct sum over the four neighbours within 1e-12, and the kernel compiles to Numba

#### Scenario: Linear exchange on a network
- **WHEN** a coupling on a network field sums `g (source.x - target.x) + xi source.x` over the neighbours
- **THEN** the compiled rate equals `g L x + xi A x` with the graph Laplacian `L` and the adjacency matrix `A` within
  1e-12

#### Scenario: Diffusive coupling on a spectral field
- **WHEN** a coupling on a spectral field sums `D (source.x - target.x)`
- **THEN** it lowers to `D laplacian(x)`, the Fourier multiplier of the field

### Requirement: Fields integrate against two-point kernels
A law SHALL integrate an expression of a field on a phase space over named class axes against a kernel `K(c, c')`
written in the coordinates of the point it updates, the coordinates of the point integrated over, and the parameters,
so the kernel need not be a function of the displacement `c - c'`. The integral SHALL use the cell widths and the cell
volumes of the space as weights, and the kernel SHALL be sampled at the cell centres when the problem is compiled.

#### Scenario: Antisymmetric kernel conserves the total
- **WHEN** a density on a class axis changes at the rate `f(z)` times the integral of `rho(z, u) - rho(u, z)` against
  `f(u)`, for the victory probability `rho(z, u) = (1 + tanh(2 (G(z) - G(u)))) / 2` of a quadratic `G`
- **THEN** the total of the density stays constant within 1e-10 relative error

#### Scenario: Kernel of the coordinates
- **WHEN** a law integrates a density that is one on `[0, 1]` against the kernel `z * u`
- **THEN** the integral at the cell of centre `z` equals `z / 2` within the midpoint error of the grid

### Requirement: Misused two-point integrals fail clearly
The compile report SHALL hold a two-point kernel that reads a state as a `kernel-state` issue, and an integral over
class axes of a field without a phase space as a `law-space` issue.

#### Scenario: Kernel of a state
- **WHEN** a law integrates a field against a kernel that reads its own state
- **THEN** the compile report holds a `kernel-state` issue that names the state

### Requirement: Fields integrate cumulatively along an axis
A law SHALL integrate an expression of a field on a lattice along one axis, from the start of the domain to every site
or from every site to the end of the domain, counting half of the cell of the site itself and weighting every cell by
its width and its volume. The result SHALL be a field of the same shape that laws read and other operators take.

#### Scenario: Tail integral of a constant
- **WHEN** a law integrates a field equal to one along a class axis of 10 cells on `[0, 1]` from every site to the end
- **THEN** the integral at the cell of centre `h` equals `1 - h`

#### Scenario: Beer-Lambert shading
- **WHEN** the light at height `h` is `exp(-gamma * integral from h to the top of rho(h') h'^2 dh')`
- **THEN** the light is one above the canopy and decreases monotonically downward through it

### Requirement: Convolutions act along the position axes of phase spaces
A convolution of a field on a phase space SHALL act along the position axes at every point of the class axes, with a
kernel of the displacements of the position axes, and it SHALL follow the boundary of the position lattice.

#### Scenario: Dispersal at every class
- **WHEN** a density over a position lattice with fixed edges of value zero and a class axis is convolved with a
  normalized exponential kernel of the position displacement
- **THEN** every class slice equals the one-dimensional convolution of that slice, and sites near the edges receive
  less than sites in the bulk
