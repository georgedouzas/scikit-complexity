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
