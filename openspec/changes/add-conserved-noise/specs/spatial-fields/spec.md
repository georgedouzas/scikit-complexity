# Spec Delta

## ADDED Requirements

### Requirement: Noise can conserve a field
A noise law on a state of a field on a lattice SHALL conserve the field when declared `conserved`: it adds the
finite-difference divergence of a random flux with one Wiener process per link, so the total of the field changes only
by its rate laws. The library SHALL provide `active_model_b` built on it.

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
