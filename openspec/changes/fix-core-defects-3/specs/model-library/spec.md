# Spec Delta

## ADDED Requirements

### Requirement: The chemical Langevin equation switches by noise
`chemical_langevin_equation` with its defaults SHALL switch between its two stable states within 200 time units.

#### Scenario: Noise-driven switching
- **WHEN** the default chemical Langevin equation runs until 200 with Euler-Maruyama steps of 1e-3 from seed 0
- **THEN** after time 20 the number of molecules is below 30 at some frame and above 90 at another

### Requirement: The hodgepodge machine forms spirals
`hodgepodge` with its defaults SHALL keep travelling waves on a 100 × 100 lattice.

#### Scenario: Waves persist
- **WHEN** the default hodgepodge machine on a 100 × 100 lattice runs for 300 steps
- **THEN** more than 10 % of the cells are ill at the last step

### Requirement: Social impact flips under net pressure
`social_impact` SHALL flip an opinion when the net pressure to change plus the random field is positive.

#### Scenario: No pressure keeps opinions
- **WHEN** a social impact model without noise starts with every agent of opinion -1
- **THEN** no opinion changes in 5 steps

### Requirement: Model defaults show the described behaviour
`restricted_three_body`, `diffusion_in_a_speckle_field`, and `legion` with their defaults SHALL show chaos, anomalous
diffusion, and alternating objects within short runs, and `burridge_knopoff` SHALL register a solver that does not
stall on its stick-slip switches.

#### Scenario: Chaotic restricted three-body orbit
- **WHEN** the default restricted three-body problem and one started 1e-8 away run until 20
- **THEN** the body passes within 0.2 of the Moon and the two paths end more than 1e-6 apart

#### Scenario: Anomalous diffusion in a speckle field
- **WHEN** the default speckle field runs until 60 in frames of 0.5
- **THEN** the particles move less than 5 units on average in the first frame, and from time 10 their mean squared
  displacement grows as `t^a` with `0.1 < a < 0.8`

#### Scenario: Both LEGION objects fire
- **WHEN** the default LEGION network runs until 100
- **THEN** each of its two objects starts an active phase

#### Scenario: Stick-slip without a stall
- **WHEN** the default Burridge-Knopoff fault runs until 400 with its registered solver
- **THEN** no warning is raised, no block moves backwards, and some frame holds a slip above 0.3
