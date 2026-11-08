## MODIFIED Requirements

### Requirement: The spectral Laplacian damps the Nyquist mode
The spectral Laplacian SHALL give every Fourier mode, the Nyquist mode included, the eigenvalue `-|k|^2`.

#### Scenario: Checkerboard mode
- **WHEN** a field `u' = u + lap(u)` on a 16 × 16 periodic lattice of unit spacing starts from a checkerboard of
  amplitude 0.01 and runs until 0.5 with the spectral discretization
- **THEN** its amplitude equals that of the finite-difference discretization within 1e-3 and decays

#### Scenario: Seeded initial fields
- **WHEN** `ComplexGinzburgLandauSystem` is built twice with the same `random_state`
- **THEN** both initial fields `u` and `v` are equal, and differ for another seed

### Requirement: Noise can conserve a field
A noise law on a state of a field on a lattice SHALL conserve the field when declared `conserved`: it adds the
finite-difference divergence of a random flux with one Wiener process per link, so the total of the field changes only
by its rate laws. The library SHALL provide `ActiveSystemBSystem` built on it.

#### Scenario: Total is conserved
- **WHEN** a field on a periodic lattice runs with only a conserved noise law for 1000 steps
- **THEN** its total stays at its initial value within 1e-9 relative, while its sites fluctuate

#### Scenario: Structure factor of conserved noise
- **WHEN** a diffusing field with conserved noise reaches its stationary state
- **THEN** its structure factor is flat in the wavenumber within 10%, the equilibrium of conserved dynamics, and
  vanishes at wavenumber zero

#### Scenario: Conserved noise off a lattice
- **WHEN** a conserved noise law targets the state of an element that is not a field on a lattice
- **THEN** validation reports `noise-unsupported` and names the lattice fields it needs
