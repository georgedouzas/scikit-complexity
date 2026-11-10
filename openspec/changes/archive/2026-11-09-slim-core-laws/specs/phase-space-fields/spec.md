## ADDED Requirements

### Requirement: Collision integrals carry their kernel
The collision integral SHALL be written as a call of the density and of its kernel `B(speed, cosine)`, so a rate law
that reads it needs no other argument. The kernel SHALL be a function that returns an expression of the relative speed,
the cosine of the deflection angle, and numbers, and a kernel that reads a state or a parameter SHALL raise a
`ValidationError` at compile time.

#### Scenario: Maxwell molecules
- **WHEN** a rate law of a density on two interval velocity axes reads `collision_integral(f, kernel)` with the constant
  kernel `1 / (2 pi)`
- **THEN** the run gives the fast spectral integral that the cross-section of a rate law gave before, within 1e-12

#### Scenario: Kernel that reads a parameter
- **WHEN** the kernel of a collision integral returns an expression of a parameter
- **THEN** compiling raises a `ValidationError` that names the law and the parameter
