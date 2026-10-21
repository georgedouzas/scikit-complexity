# Spec Delta

## ADDED Requirements

### Requirement: Specific realizations live in the model library
The library SHALL hold every specific realization of a core primitive in the module of the field and subfield that
introduced it, and other fields SHALL import it from there. The realizations the core holds today SHALL move to
`economics`, `social`, `physics`, `mathematics`, and `engineering`, as the design's table of moved realizations lists.

#### Scenario: Order book from the library
- **WHEN** a user imports the order book from `skcomplex.models.economics`
- **THEN** it is a pairing function for the core's matching law, and its priority scenario holds: the earlier of two
  same-price sell orders trades first

#### Scenario: Potential from the library
- **WHEN** a user imports the Lennard-Jones potential from `skcomplex.models.physics`
- **THEN** it is a realization of the core's potential primitive, and the existing LAMMPS agreement tests pass with it

### Requirement: Library models keep their behaviour across the redesign
Every library model SHALL produce the same trajectory on fixed seeds after migrating to the new vocabulary, within the
tolerances of the conformance suite.

#### Scenario: Conformance before and after
- **WHEN** the conformance suite runs every catalog model before and after the migration with random state 0
- **THEN** every trajectory agrees within its declared tolerance
