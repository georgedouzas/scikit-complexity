# Spec Delta

## Purpose

Lets a user declare complex-valued states and parameters on elements, groups, and fields, write laws over them with
complex arithmetic and functions, and simulate, record, analyse, and visualise them with the same solvers as real
systems.

## ADDED Requirements

### Requirement: States and parameters take a complex domain
The modelling layer SHALL accept a complex domain for states and parameters of single elements, groups, and fields.
Their values SHALL be any finite complex number, and real values SHALL be accepted and stored with a zero imaginary
part. A real domain SHALL reject a value with a nonzero imaginary part.

#### Scenario: Complex initial value
- **WHEN** a user declares a complex state `z` with default `1 + 2j` and builds the element
- **THEN** the compiled initial state reads back `z` as `1 + 2j`

#### Scenario: Complex value in a real domain
- **WHEN** a user gives the value `1j` to a real state
- **THEN** building the element raises a named error that names the state, its domain, and the complex domain to
  declare instead

#### Scenario: Complex parameter per site
- **WHEN** a field on a 4 × 4 lattice takes a complex parameter `c` with one value per site
- **THEN** every site reads its own value of `c`, and changing `c` while running changes the next step

### Requirement: Laws use complex arithmetic and functions
Laws SHALL use the imaginary unit, the real part, imaginary part, conjugate, modulus, and argument, and the
exponential, logarithm, square root, trigonometric, and hyperbolic functions over the complex plane. A law on a real
state SHALL take a real value, and the compiler SHALL reject a law whose value is complex for a real state. A
relation that orders values SHALL take real operands.

#### Scenario: Harmonic phase
- **WHEN** a complex state `z` with `z(0) = 1` follows `Rate(z, i * omega * z)` with `omega = 2` until time 10
- **THEN** `z(10)` equals `exp(20 i)` within 1e-8, and the modulus stays 1 within 1e-8 at every frame

#### Scenario: Real readout of a complex state
- **WHEN** a real state `r` follows `Update(r, abs(z))` for a complex state `z`
- **THEN** the system compiles, and `r` holds the modulus of `z` after every step

#### Scenario: Complex value for a real state
- **WHEN** a real state `x` follows `Rate(x, i * x)`
- **THEN** compiling raises a named error that names the law and the state, and suggests the real part, the imaginary
  part, or a complex domain

#### Scenario: Ordering a complex value
- **WHEN** a law uses the condition `z > 2` for a complex state `z`
- **THEN** compiling raises a named error that names the law and suggests comparing the modulus or a real part

### Requirement: Complex laws compile to every backend
Complex laws SHALL compile to the NumPy backend and to the Numba backend, inside and outside groups and fields, and
both backends SHALL give the same results.

#### Scenario: Backends agree on a complex map
- **WHEN** a group of 10000 members follows `Update(z, z**2 + c)` with complex `z` and `c`, and one step is evaluated
  with both backends from the same state
- **THEN** the results agree within 1e-12 relative error

#### Scenario: Complex laws stay cached
- **WHEN** the same complex system is compiled with the Numba backend in two separate processes
- **THEN** the second compilation reuses the cached kernels

### Requirement: Complex systems run with the existing solvers
Every solver and stepper that runs a real system of the same form SHALL run a complex system, including the
default ODE solver, the fixed-step steppers, the stiff and exponential steppers, and spectral fields. A complex
system SHALL give the same trajectory as the system written with the real and imaginary parts as separate real
states.

#### Scenario: Complex and split forms agree
- **WHEN** the complex Ginzburg-Landau equation is written once with one complex field and once with two real fields,
  and both run spectrally with the same exponential stepper, step, and initial amplitude
- **THEN** the complex trajectory equals `u + i v` of the split trajectory within 1e-10 at every frame

#### Scenario: Unitary evolution
- **WHEN** the free Schrödinger equation `Rate(psi, i / 2 * laplacian(psi))` runs spectrally on a periodic lattice of
  256 sites from a Gaussian wave packet until time 5
- **THEN** the norm `sum(abs(psi)**2)` stays constant within 1e-8 relative error

### Requirement: Trajectories return complex values
Trajectories, frames, and observables SHALL return complex arrays for complex variables and real arrays for real
variables. Analysis functions that require real values SHALL reject complex input with a named error that names the
variable and the real views to use: real part, imaginary part, modulus, or argument.

#### Scenario: Complex trajectory
- **WHEN** a trajectory of a field with a complex state on a 32 × 32 lattice is asked for that state as a lattice
- **THEN** it returns a complex array of shape `(frames, 32, 32)`

#### Scenario: Real analysis of a complex series
- **WHEN** a user asks for the power spectrum of a complex state's series
- **THEN** a named error names the state and the real views to take first

### Requirement: Complex values stream as modulus and phase
Visuals SHALL stream complex states to Rerun as their modulus and their phase. A complex lattice field SHALL appear as
two images, one colored by modulus with the declared range and one colored by phase with a cyclic color map over
`(-π, π]`. A complex scalar state SHALL appear as two series, its real and imaginary parts.

#### Scenario: Complex field in the viewer
- **WHEN** a complex field `A` of an element at path `medium` streams to Rerun
- **THEN** the recording holds images under `medium/A/modulus` and `medium/A/phase` for every logged frame

#### Scenario: Complex scalar in the viewer
- **WHEN** a single element with a complex state `z` streams to Rerun
- **THEN** the recording holds scalar series under `<path>/z/real` and `<path>/z/imag`
