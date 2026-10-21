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
- **WHEN** a Bak-Tang-Wiesenfeld sandpile on a 32 × 32 lattice drops 5000 grains from a fixed seed, once with an
  activity and once without
- **THEN** both runs end with the same heights and record avalanches at the same times with the same durations

#### Scenario: Fewer sites evaluated
- **WHEN** the same two runs count the sites their toppling rule evaluates
- **THEN** the run with an activity evaluates fewer than a tenth of the sites of the run without
