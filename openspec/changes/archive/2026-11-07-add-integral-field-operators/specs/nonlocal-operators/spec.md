# Spec Delta

## ADDED Requirements

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
