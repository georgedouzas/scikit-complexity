## ADDED Requirements

### Requirement: Field measures are realizations of general analyses
A measure named by one field, such as the Binder cumulant, the polar order, the kinetic temperature, or the
dissimilarity index, SHALL be a realization in that field's package: a subclass of a general analysis that only fixes
settings in its constructor and documents the field's name, adding no formula, verb, or attribute. Its package SHALL
follow the placement rule of realizations.

#### Scenario: Binder cumulant of a model
- **WHEN** a user runs `BinderCumulantAnalysis` from the physics realizations on magnetisation samples
- **THEN** it sets the same attributes as the general distribution analysis of kind `moment-ratio` with orders 4 and 2,
  with the same values

#### Scenario: Polar order of a flock
- **WHEN** a user runs `PolarOrderAnalysis` on a Vicsek trajectory
- **THEN** it equals the alignment analysis of kind `mean-vector` on the headings of the flock
