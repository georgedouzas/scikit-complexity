# Spec Delta

## ADDED Requirements

### Requirement: Lattice steps apply boundary laws
Lattice and Monte Carlo steppers SHALL apply the boundary laws of the problem, including the wrapping of angles, after
every step.

#### Scenario: XY angles stay wrapped
- **WHEN** the XY model on a 16 × 16 lattice runs until 50 with Metropolis rotations
- **THEN** every recorded angle lies in `(-pi, pi]`

### Requirement: Relaxations update their active neighbourhoods
A relaxation that declares its active sites SHALL update, in every iteration after the first, only the active sites
and the sites that read them, and SHALL record the same avalanches as a full update.

#### Scenario: Same avalanches as a full update
- **WHEN** the Bak-Tang-Wiesenfeld sandpile on a 32 × 32 lattice drops 3000 grains from a fixed seed
- **THEN** its final heights and the sizes, durations, and areas of its avalanches equal those of a full update of
  every site in every iteration

#### Scenario: Steady-state speed
- **WHEN** the Bak-Tang-Wiesenfeld sandpile on a 64 × 64 lattice drops 2000 grains after 5000 grains of transient
- **THEN** the 2000 grains take less than 2 seconds
