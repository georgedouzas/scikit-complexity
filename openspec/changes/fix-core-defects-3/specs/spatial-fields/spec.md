# Spec Delta

## ADDED Requirements

### Requirement: The spectral Laplacian damps the Nyquist mode
The spectral Laplacian SHALL give every Fourier mode, the Nyquist mode included, the eigenvalue `-|k|^2`.

#### Scenario: Checkerboard mode
- **WHEN** a field `u' = u + lap(u)` on a 16 × 16 periodic lattice of unit spacing starts from a checkerboard of
  amplitude 0.01 and runs until 0.5 with the spectral discretization
- **THEN** its amplitude equals that of the finite-difference discretization within 1e-3 and decays

#### Scenario: Seeded initial fields
- **WHEN** `complex_ginzburg_landau` is built twice with the same `random_state`
- **THEN** both initial fields `u` and `v` are equal, and differ for another seed
